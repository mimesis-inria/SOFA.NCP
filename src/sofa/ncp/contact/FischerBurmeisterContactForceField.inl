/****************************************************************************
* Generic Fischer-Burmeister NCP contact force field implementation.
****************************************************************************/
#pragma once

#include <sofa/ncp/contact/FischerBurmeisterContactForceField.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/logging/Messaging.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace sofa::ncp
{

template<class T1, class T2>
FischerBurmeisterContactForceField<T1, T2>::FischerBurmeisterContactForceField()
    : Inherit()
    , d_fixedComplianceScale(initData(&d_fixedComplianceScale, Real(1.0),
          "fixedComplianceScale", "Positive fallback r used in phi(g,r*lambda)."))
    , d_complianceMode(initData(&d_complianceMode, static_cast<unsigned int>(Fixed),
          "complianceMode", "0=fixed, 1=lagged reference-elastic scale, 2=current scale (reserved)."))
    , d_fbEpsilon(initData(&d_fbEpsilon, Real(0),
          "fbEpsilon", "Nonnegative Fischer-Burmeister smoothing epsilon."))
    , d_contactNewtonRegularization(initData(&d_contactNewtonRegularization, Real(0.0),
          "contactNewtonRegularization",
          "Nonnegative tangent-only diagonal regularization added to active lambda rows."))
    , l_beamForceField(initLink("beamForceField",
          "BeamFEMForceField providing the constant positive reference elastic metric."))
    , l_fixedConstraint(initLink("fixedConstraint",
          "Fixed projective constraint eliminated from the reference elastic metric."))
    , d_showContactGradients(initData(&d_showContactGradients, true,
          "showContactGradients", "Draw active near-contact gap gradients."))
    , d_drawGradientScale(initData(&d_drawGradientScale, Real(5),
          "drawGradientScale", "Contact-gradient drawing scale."))
    , d_contactColor(initData(&d_contactColor, sofa::type::RGBAColor(1.0f, 0.85f, 0.0f, 1.0f),
          "contactColor", "Contact-gradient drawing color."))
    , d_debug(initData(&d_debug, true, "debug", "Publish compact contact/compliance diagnostics."))
    , d_publishContactData(initData(&d_publishContactData, true,
          "publishContactData", "Publish point-indexed contact diagnostics."))
    , d_contactStatus(initData(&d_contactStatus, "contactStatus", "0=Active, 1=Pinned, 2=InvalidGeometry."))
    , d_contactGapGradient(initData(&d_contactGapGradient, "contactGapGradient", "Point-indexed H=dg/dx."))
    , d_contactGap(initData(&d_contactGap, "contactGap", "Point-indexed gaps."))
    , d_contactLambda(initData(&d_contactLambda, "contactLambda", "Point-indexed multipliers."))
    , d_contactComplianceScale(initData(&d_contactComplianceScale, "contactComplianceScale", "Point-indexed scalar r."))
    , d_contactScaledLambda(initData(&d_contactScaledLambda, "contactScaledLambda", "Point-indexed r*lambda."))
    , d_contactPhi(initData(&d_contactPhi, "contactPhi", "Point-indexed row residuals."))
    , d_contactDPhiDgap(initData(&d_contactDPhiDgap, "contactDPhiDgap", "Point-indexed dphi/dg."))
    , d_contactDPhiDlambda(initData(&d_contactDPhiDlambda, "contactDPhiDlambda", "Point-indexed dphi/dlambda."))
    , d_activeContactCount(initData(&d_activeContactCount, sofa::Size(0), "activeContactCount", "Active FB rows."))
    , d_pinnedContactCount(initData(&d_pinnedContactCount, sofa::Size(0), "pinnedContactCount", "Pinned rows."))
    , d_invalidContactCount(initData(&d_invalidContactCount, sofa::Size(0), "invalidContactCount", "Invalid rows."))
    , d_referenceDelassus(initData(&d_referenceDelassus,
          "referenceDelassus",
          "Flattened row-major active-contact W_ref = H K_ref^{-1} H^T."))
    , d_referenceDelassusLambdaIndices(initData(&d_referenceDelassusLambdaIndices,
        "referenceDelassusLambdaIndices",
        "Lambda MechanicalState index associated with each W_ref row/column."))
    , d_referenceDelassusSize(initData(&d_referenceDelassusSize, sofa::Size(0),
        "referenceDelassusSize",
        "Number of active rows/columns in referenceDelassus."))
{
    static_assert(T1::spatial_dimensions >= TranslationalDim,
        "Object1 must expose at least three translational DOFs.");
    static_assert(T2::spatial_dimensions == 1,
        "Object2 must be a scalar Vec1-like multiplier state.");

    d_contactStatus.setReadOnly(true);
    d_contactGapGradient.setReadOnly(true);
    d_contactGap.setReadOnly(true);
    d_contactLambda.setReadOnly(true);
    d_contactComplianceScale.setReadOnly(true);
    d_contactScaledLambda.setReadOnly(true);
    d_contactPhi.setReadOnly(true);
    d_contactDPhiDgap.setReadOnly(true);
    d_contactDPhiDlambda.setReadOnly(true);
    d_activeContactCount.setReadOnly(true);
    d_pinnedContactCount.setReadOnly(true);
    d_invalidContactCount.setReadOnly(true);
    d_referenceDelassus.setReadOnly(true);
    d_referenceDelassusLambdaIndices.setReadOnly(true);
    d_referenceDelassusSize.setReadOnly(true);
}

template<class T1, class T2>
FischerBurmeisterContactForceField<T1, T2>::FischerBurmeisterContactForceField(
    core::behavior::MechanicalState<DataTypes1>* object1,
    core::behavior::MechanicalState<DataTypes2>* object2)
    : FischerBurmeisterContactForceField()
{
    this->mstate1.set(object1);
    this->mstate2.set(object2);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::initializeContactRows()
{
    if (!this->mstate1 || !this->mstate2)
    {
        msg_error() << "object1 and object2 MechanicalState links are required.";
        m_validState = false;
        return false;
    }

    if (this->mstate1->getSize() != this->mstate2->getSize())
    {
        msg_error() << "object1 size (" << this->mstate1->getSize()
                    << ") and lambda size (" << this->mstate2->getSize()
                    << ") must match.";
        m_validState = false;
        return false;
    }

    const unsigned int mode = d_complianceMode.getValue();
    if (mode > static_cast<unsigned int>(Current))
    {
        msg_error() << "complianceMode must be 0=fixed, 1=lagged or 2=current.";
        m_validState = false;
        return false;
    }

    if (mode == static_cast<unsigned int>(Current))
    {
        msg_error() << "complianceMode=2 is reserved. Same-linearization compliance requires a base-residual refresh.";
        m_validState = false;
        return false;
    }

    if (!(d_fixedComplianceScale.getValue() > Real(0))
        || !std::isfinite(static_cast<double>(d_fixedComplianceScale.getValue())))
    {
        msg_error() << "fixedComplianceScale must be finite and positive.";
        m_validState = false;
        return false;
    }

    if (!(d_fbEpsilon.getValue() >= Real(0))
        || !std::isfinite(static_cast<double>(d_fbEpsilon.getValue())))
    {
        msg_error() << "fbEpsilon must be finite and nonnegative.";
        m_validState = false;
        return false;
    }

    if (!(d_contactNewtonRegularization.getValue() >= Real(0))
        || !std::isfinite(static_cast<double>(d_contactNewtonRegularization.getValue())))
    {
        msg_error() << "contactNewtonRegularization must be finite and nonnegative.";
        m_validState = false;
        return false;
    }

    m_contacts.clear();
    m_validState = false;
    return true;
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::init()
{
    Inherit::init();

    if (!l_beamForceField)
    {
        l_beamForceField.set(this->getContext()->template get<BeamForceField>(
            this->getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));
    }

    if (!l_fixedConstraint)
    {
        l_fixedConstraint.set(this->getContext()->template get<FixedConstraint>(
            this->getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));
    }

    const bool valid = initializeContactRows();

    if (valid && usesLaggedCompliance())
    {
        if constexpr (!std::is_same_v<DataTypes1, sofa::defaulttype::Rigid3Types>)
        {
            msg_error() << "complianceMode=1 currently requires object1=Rigid3.";
            this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
            return;
        }

        if (!l_beamForceField)
        {
            msg_error() << "complianceMode=1 requires a BeamFEMForceField link.";
            this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
            return;
        }

        if (!l_fixedConstraint)
        {
            msg_error() << "complianceMode=1 requires a FixedProjectiveConstraint link.";
            this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    this->d_componentState.setValue(valid
        ? core::objectmodel::ComponentState::Valid
        : core::objectmodel::ComponentState::Invalid);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::reinit()
{
    Inherit::reinit();

    // Keep the cached metric/scales when beam K0 and constrained indices did not change.
    // beginNonlinearSolve() checks both cache signatures before the next promotion.
    const bool valid = initializeContactRows();

    this->d_componentState.setValue(valid
        ? core::objectmodel::ComponentState::Valid
        : core::objectmodel::ComponentState::Invalid);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::usesLaggedCompliance() const
{
    return d_complianceMode.getValue() == static_cast<unsigned int>(Lagged);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::usesCurrentCompliance() const
{
    return d_complianceMode.getValue() == static_cast<unsigned int>(Current);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::beginNonlinearSolve()
{
    if (!usesLaggedCompliance())
        return;

    if (!ensureReferenceComplianceCache())
    {
        msg_warning() << "[NCP SCALE] Reference elastic metric unavailable; keeping current/fixed r.";
        return;
    }

    const sofa::Size expectedPoints = this->mstate1 ? this->mstate1->getSize() : 0;

    if (m_hasNextCompliance && m_nextCompliance.size() == expectedPoints)
    {
        m_currentCompliance = std::move(m_nextCompliance);
        m_nextCompliance.clear();
        m_hasNextCompliance = false;
        m_hasCurrentCompliance = true;
        ++m_complianceGeneration;
    }

    if (m_hasCurrentCompliance && m_currentCompliance.size() != expectedPoints)
    {
        m_currentCompliance.clear();
        m_hasCurrentCompliance = false;
    }
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::discardLaggedComplianceSnapshot()
{
    m_nextCompliance.clear();
    m_hasNextCompliance = false;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::Real
FischerBurmeisterContactForceField<T1, T2>::positiveOrFallback(Real value, Real fallback)
{
    return std::isfinite(static_cast<double>(value)) && value > Real(0) ? value : fallback;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::Vec3
FischerBurmeisterContactForceField<T1, T2>::extractPosition(const Coord1& coordinate)
{
    const auto p = DataTypes1::getCPos(coordinate);
    return Vec3(static_cast<Real>(p[0]), static_cast<Real>(p[1]), static_cast<Real>(p[2]));
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::computeGapHessian(const Vec3&, Mat3& hessian) const
{
    hessian.clear();
    return false;
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::hasValidKinematics(const Contact& c)
{
    const Real norm2 = c.gapGradient.norm2();
    return std::isfinite(static_cast<double>(c.gap))
        && std::isfinite(static_cast<double>(c.lambda))
        && std::isfinite(static_cast<double>(c.gapGradient[0]))
        && std::isfinite(static_cast<double>(c.gapGradient[1]))
        && std::isfinite(static_cast<double>(c.gapGradient[2]))
        && std::isfinite(static_cast<double>(norm2))
        && norm2 > Real(1e-30);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::updateFischerBurmeisterTerms(Contact& c) const
{
    const Real eps = d_fbEpsilon.getValue();
    const Real r = c.complianceScale;
    const Real contactCompliance = 0;

    const Real effectiveGap = c.gap + contactCompliance * c.lambda;
    const Real s = r * c.lambda;
    const Real n = std::sqrt(effectiveGap * effectiveGap+ s * s+ eps * eps);

    c.scaledLambda = s;

    const Real phi = effectiveGap + s - n;

    // d phi / d g_eff
    const Real a = Real(1) - effectiveGap / n;

    // Contribution from s = r * lambda.
    const Real b = r * (Real(1) - s / n);
    const Real invR = Real(1) / r;

    c.phi = phi * invR;
    c.dPhiDgap = a * invR;
    c.dPhiDlambda = (a * contactCompliance + b) * invR;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::Real
FischerBurmeisterContactForceField<T1, T2>::complianceForPoint(sofa::Index pointIndex, Real fallback) const
{
    if (!usesLaggedCompliance() || !m_hasCurrentCompliance || pointIndex >= m_currentCompliance.size())
        return fallback;

    return positiveOrFallback(m_currentCompliance[pointIndex], fallback);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::finalizeContactRow(Contact& c, ContactStatus geometryStatus, Real fixedR) const
{
    if (geometryStatus == ContactStatus::Pinned || c.gap >= 0.5)
    {
        c.status = ContactStatus::Pinned;
        c.gap = Real(0);
        c.gapGradient.clear();
        c.complianceScale = Real(1);
        c.scaledLambda = c.lambda;
        c.phi = c.lambda;
        c.dPhiDgap = Real(0);
        c.dPhiDlambda = Real(1);
        return;
    }

    if (geometryStatus != ContactStatus::Active || !hasValidKinematics(c))
    {
        c.status = ContactStatus::InvalidGeometry;
        c.gap = Real(0);
        c.gapGradient.clear();
        c.complianceScale = Real(1);
        c.scaledLambda = c.lambda;
        c.phi = c.lambda;
        c.dPhiDgap = Real(0);
        c.dPhiDlambda = Real(1);
        return;
    }

    c.status = ContactStatus::Active;
    c.complianceScale = complianceForPoint(c.pointIndex, fixedR);
    updateFischerBurmeisterTerms(c);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::rebuildCurrentContacts(const VecCoord1& x1, const VecCoord2& x2)
{
    if (x1.size() != x2.size() || x1.empty())
    {
        msg_error() << "Cannot build contact rows: object1 has " << x1.size()
                    << " entries and object2 has " << x2.size() << ".";
        m_contacts.clear();
        m_validState = false;
        return false;
    }

    if (m_contacts.size() != x1.size())
        m_contacts.resize(x1.size());

    const Real fixedR = d_fixedComplianceScale.getValue();
    bool valid = true;

    for (sofa::Index i = 0; i < x1.size(); ++i)
    {
        Contact& c = m_contacts[i];
        c = Contact{};
        c.pointIndex = i;
        c.lambdaIndex = i;
        c.lambda = static_cast<Real>(x2[i][0]);

        const ContactStatus status = computeContactKinematics(extractPosition(x1[i]), c);
        finalizeContactRow(c, status, fixedR);
        valid = valid && c.status != ContactStatus::InvalidGeometry;
    }

    m_validState = valid;
    publishDebugData();
    return valid;
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::clearReferenceDelassus()
{
    m_referenceDelassus.clear();
    m_referenceDelassusLambdaIndices.clear();

    d_referenceDelassus.setValue(sofa::type::vector<Real>{});
    d_referenceDelassusLambdaIndices.setValue(sofa::type::vector<unsigned int>{});
    d_referenceDelassusSize.setValue(sofa::Size(0));
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::publishReferenceDelassus()
{
    d_referenceDelassus.setValue(m_referenceDelassus);
    d_referenceDelassusSize.setValue(m_referenceDelassusLambdaIndices.size());

    sofa::type::vector<unsigned int> indices;
    indices.reserve(m_referenceDelassusLambdaIndices.size());

    for (const sofa::Index index : m_referenceDelassusLambdaIndices)
        indices.push_back(static_cast<unsigned int>(index));

    d_referenceDelassusLambdaIndices.setValue(indices);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::getReferenceTranslationalComplianceBlock(
    sofa::Index pointI, sofa::Index pointJ, Mat3& block) const
{
    block.clear();

    if (!m_referenceComplianceCacheValid
        || pointI >= m_referenceCompliancePointCount
        || pointJ >= m_referenceCompliancePointCount)
    {
        return false;
    }

    const std::size_t index =
        static_cast<std::size_t>(pointI) * m_referenceCompliancePointCount
        + static_cast<std::size_t>(pointJ);

    if (index >= m_referenceTranslationalComplianceBlocks.size())
        return false;

    block = m_referenceTranslationalComplianceBlocks[index];
    return true;
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::invalidateReferenceComplianceCache()
{
    m_referenceTranslationalComplianceBlocks.clear();
    m_referenceCompliancePointCount = 0;
    clearReferenceDelassus();

    m_currentCompliance.clear();
    m_nextCompliance.clear();
    m_hasCurrentCompliance = false;
    m_hasNextCompliance = false;

    m_cachedReferenceMetricVersion = std::numeric_limits<sofa::Size>::max();
    m_cachedConstraintSignature = 0;
    m_referenceComplianceCacheValid = false;
}

template<class T1, class T2>
std::size_t FischerBurmeisterContactForceField<T1, T2>::fixedConstraintSignature() const
{
    const auto* constraint = l_fixedConstraint.get();
    if (!constraint)
        return 0;

    std::size_t hash = static_cast<std::size_t>(1469598103934665603ULL);

    auto mix = [&hash](std::size_t value)
    {
        hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2);
    };

    mix(constraint->fixAllDOFs() ? 1u : 0u);

    for (const sofa::Index point : constraint->d_indices.getValue())
        mix(static_cast<std::size_t>(point));

    return hash;
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::ensureReferenceComplianceCache()
{
    if (!this->mstate1 || !l_beamForceField || !l_fixedConstraint)
        return false;

    const auto* beam = l_beamForceField.get();
    const sofa::Size version = beam->getReferenceElasticMetricVersion();
    const std::size_t constraintSignature = fixedConstraintSignature();
    const sofa::Size pointCount = this->mstate1->getSize();
    const std::size_t blockCount =
        static_cast<std::size_t>(pointCount) * static_cast<std::size_t>(pointCount);

    if (m_referenceComplianceCacheValid
        && m_cachedReferenceMetricVersion == version
        && m_cachedConstraintSignature == constraintSignature
        && m_referenceCompliancePointCount == pointCount
        && m_referenceTranslationalComplianceBlocks.size() == blockCount)
    {
        return true;
    }

    return rebuildReferenceComplianceCache();
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::rebuildReferenceComplianceCache()
{
    if constexpr (!std::is_same_v<DataTypes1, sofa::defaulttype::Rigid3Types>)
    {
        return false;
    }
    else
    {
        auto* beam = l_beamForceField.get();
        auto* constraint = l_fixedConstraint.get();

        if (!beam || !constraint || !this->mstate1)
            return false;

        invalidateReferenceComplianceCache();

        if (constraint->fixAllDOFs())
        {
            msg_error() << "[NCP SCALE] All mechanical points are fixed.";
            return false;
        }

        const sofa::Size pointCount = this->mstate1->getSize();
        const sofa::Size objectDofs = pointCount * DerivDim1;

        if (pointCount == 0 || DerivDim1 != 6)
            return false;

        sofa::type::vector<bool> constrainedPoint(pointCount, false);

        for (const sofa::Index point : constraint->d_indices.getValue())
        {
            if (point >= pointCount)
            {
                msg_warning() << "[NCP SCALE] Ignoring constrained point=" << point
                              << " pointCount=" << pointCount;
                continue;
            }

            constrainedPoint[point] = true;
        }

        sofa::type::vector<int> freeIndex(objectDofs, -1);
        int freeDofCount = 0;

        for (sofa::Index point = 0; point < pointCount; ++point)
        {
            if (constrainedPoint[point])
                continue;

            for (sofa::Size d = 0; d < DerivDim1; ++d)
                freeIndex[point * DerivDim1 + d] = freeDofCount++;
        }

        if (freeDofCount == 0)
            return false;

        using SparseMatrix = Eigen::SparseMatrix<Real>;
        using Triplet = Eigen::Triplet<Real>;
        using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

        const sofa::Size elementCount = beam->getReferenceElasticMetricElementCount();
        std::vector<Triplet> triplets;
        triplets.reserve(static_cast<std::size_t>(elementCount) * 144u);

        typename BeamForceField::StiffnessMatrix Ke;

        for (sofa::Size element = 0; element < elementCount; ++element)
        {
            sofa::Index a = 0;
            sofa::Index b = 0;

            if (!beam->getReferenceElasticMetricElement(element, a, b, Ke))
                return false;

            if (a >= pointCount || b >= pointCount)
            {
                msg_error() << "[NCP SCALE] Beam metric element references invalid node "
                            << a << "," << b << " pointCount=" << pointCount;
                return false;
            }

            sofa::Index global[12];

            for (sofa::Size d = 0; d < 6; ++d)
            {
                global[d] = a * 6 + d;
                global[6 + d] = b * 6 + d;
            }

            for (sofa::Size row = 0; row < 12; ++row)
            {
                const int reducedRow = freeIndex[global[row]];
                if (reducedRow < 0)
                    continue;

                for (sofa::Size col = 0; col < 12; ++col)
                {
                    const int reducedCol = freeIndex[global[col]];
                    if (reducedCol < 0)
                        continue;

                    const Real value = Real(0.5) * (Ke(row, col) + Ke(col, row));
                    if (value != Real(0))
                        triplets.emplace_back(reducedRow, reducedCol, value);
                }
            }
        }

        SparseMatrix Kfree(freeDofCount, freeDofCount);
        Kfree.setFromTriplets(
            triplets.begin(), triplets.end(),
            [](const Real& a, const Real& b) { return a + b; });
        Kfree.makeCompressed();

        Eigen::SimplicialLDLT<SparseMatrix> factorization;
        factorization.compute(Kfree);

        if (factorization.info() != Eigen::Success)
        {
            msg_error() << "[NCP SCALE] Reference elastic metric factorization failed.";
            return false;
        }

        const auto D = factorization.vectorD();
        Real minD = std::numeric_limits<Real>::infinity();
        Real maxD = Real(0);

        for (Eigen::Index i = 0; i < D.size(); ++i)
        {
            const Real value = D[i];

            if (!std::isfinite(static_cast<double>(value)) || value <= Real(0))
            {
                msg_error() << "[NCP SCALE] Reference elastic metric is not positive definite."
                            << " D[" << i << "]=" << value;
                return false;
            }

            minD = std::min(minD, value);
            maxD = std::max(maxD, value);
        }

        sofa::type::vector<int> basisColumn(pointCount * TranslationalDim, -1);
        int basisCount = 0;

        for (sofa::Index point = 0; point < pointCount; ++point)
        {
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
            {
                if (freeIndex[point * DerivDim1 + d] >= 0)
                    basisColumn[point * TranslationalDim + d] = basisCount++;
            }
        }

        DenseMatrix rhs(freeDofCount, basisCount);
        rhs.setZero();

        for (sofa::Index point = 0; point < pointCount; ++point)
        {
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
            {
                const int reduced = freeIndex[point * DerivDim1 + d];
                const int column = basisColumn[point * TranslationalDim + d];

                if (reduced >= 0 && column >= 0)
                    rhs(reduced, column) = Real(1);
            }
        }

        const DenseMatrix response = factorization.solve(rhs);

        if (factorization.info() != Eigen::Success || !response.allFinite())
        {
            msg_error() << "[NCP SCALE] Reference elastic metric solve failed.";
            return false;
        }

        const std::size_t blockCount =
            static_cast<std::size_t>(pointCount) * static_cast<std::size_t>(pointCount);

        sofa::type::vector<Mat3> blocks(blockCount);

        // K_ref is SPD, hence Ctt_ji = Ctt_ij^T. Extract only one triangular
        // half from the multi-RHS solve and mirror it exactly.
        for (sofa::Index pointI = 0; pointI < pointCount; ++pointI)
        {
            for (sofa::Index pointJ = pointI; pointJ < pointCount; ++pointJ)
            {
                Mat3& Cij = blocks[
                    static_cast<std::size_t>(pointI) * pointCount
                    + static_cast<std::size_t>(pointJ)];
                Cij.clear();

                for (sofa::Size row = 0; row < TranslationalDim; ++row)
                {
                    const int reducedRow = freeIndex[pointI * DerivDim1 + row];
                    if (reducedRow < 0)
                        continue;

                    for (sofa::Size col = 0; col < TranslationalDim; ++col)
                    {
                        const int column = basisColumn[pointJ * TranslationalDim + col];
                        if (column >= 0)
                            Cij(row, col) = response(reducedRow, column);
                    }
                }

                Mat3& Cji = blocks[
                    static_cast<std::size_t>(pointJ) * pointCount
                    + static_cast<std::size_t>(pointI)];

                if (pointI == pointJ)
                {
                    for (sofa::Size row = 0; row < TranslationalDim; ++row)
                    {
                        for (sofa::Size col = row + 1; col < TranslationalDim; ++col)
                        {
                            const Real value = Real(0.5) * (Cij(row, col) + Cij(col, row));
                            Cij(row, col) = value;
                            Cij(col, row) = value;
                        }
                    }
                }
                else
                {
                    for (sofa::Size row = 0; row < TranslationalDim; ++row)
                        for (sofa::Size col = 0; col < TranslationalDim; ++col)
                            Cji(col, row) = Cij(row, col);
                }
            }
        }

        m_referenceTranslationalComplianceBlocks = std::move(blocks);
        m_referenceCompliancePointCount = pointCount;
        m_cachedReferenceMetricVersion = beam->getReferenceElasticMetricVersion();
        m_cachedConstraintSignature = fixedConstraintSignature();
        m_referenceComplianceCacheValid = true;

        return true;
    }
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::computeLaggedComplianceFromCurrentContacts(
    sofa::type::vector<Real>& candidate)
{
    if (!m_referenceComplianceCacheValid || !this->mstate1)
        return false;

    const sofa::Size pointCount = this->mstate1->getSize();
    const std::size_t blockCount =
        static_cast<std::size_t>(pointCount) * static_cast<std::size_t>(pointCount);

    if (m_referenceCompliancePointCount != pointCount
        || m_referenceTranslationalComplianceBlocks.size() != blockCount)
    {
        return false;
    }

    const Real fallback = d_fixedComplianceScale.getValue();

    if (m_hasCurrentCompliance && m_currentCompliance.size() == pointCount)
        candidate = m_currentCompliance;
    else
        candidate.assign(pointCount, fallback);

    sofa::type::vector<sofa::Index> active;
    active.reserve(m_contacts.size());

    for (sofa::Index i = 0; i < m_contacts.size(); ++i)
    {
        if (m_contacts[i].status == ContactStatus::Active)
            active.push_back(i);
    }

    const sofa::Size activeCount = active.size();
    m_referenceDelassus.assign(
        static_cast<std::size_t>(activeCount) * static_cast<std::size_t>(activeCount),
        Real(0));
    m_referenceDelassusLambdaIndices.resize(activeCount);

    for (sofa::Size i = 0; i < activeCount; ++i)
        m_referenceDelassusLambdaIndices[i] = m_contacts[active[i]].lambdaIndex;

    Real minR = std::numeric_limits<Real>::infinity();
    Real maxR = Real(0);
    Real sumR = Real(0);
    sofa::Size validScaleCount = 0;

    for (sofa::Size row = 0; row < activeCount; ++row)
    {
        const Contact& ci = m_contacts[active[row]];

        if (ci.pointIndex >= pointCount)
        {
            clearReferenceDelassus();
            return false;
        }

        for (sofa::Size col = row; col < activeCount; ++col)
        {
            const Contact& cj = m_contacts[active[col]];

            if (cj.pointIndex >= pointCount)
            {
                clearReferenceDelassus();
                return false;
            }

            const Mat3& Cij =
                m_referenceTranslationalComplianceBlocks[
                    static_cast<std::size_t>(ci.pointIndex) * pointCount
                    + static_cast<std::size_t>(cj.pointIndex)];

            Vec3 CijHj;
            CijHj.clear();

            for (sofa::Size i = 0; i < TranslationalDim; ++i)
                for (sofa::Size j = 0; j < TranslationalDim; ++j)
                    CijHj[i] += Cij(i, j) * cj.gapGradient[j];

            const Real wij = ci.gapGradient * CijHj;

            if (!std::isfinite(static_cast<double>(wij)))
            {
                msg_warning() << "[NCP DELASSUS] Non-finite entry"
                              << " lambdaI=" << ci.lambdaIndex
                              << " lambdaJ=" << cj.lambdaIndex;
                clearReferenceDelassus();
                return false;
            }

            m_referenceDelassus[static_cast<std::size_t>(row) * activeCount + col] = wij;
            m_referenceDelassus[static_cast<std::size_t>(col) * activeCount + row] = wij;
        }

        // Real rowNorm2 = Real(0);

        // for (sofa::Size col = 0; col < activeCount; ++col)
        // {
        //     const Real wij =m_referenceDelassus[static_cast<std::size_t>(row) * activeCount + col];
        //     rowNorm2 += wij * wij;
        // }

        const Real r = m_referenceDelassus[static_cast<std::size_t>(row) * activeCount + row];

        if (std::isfinite(static_cast<double>(r)) && r > Real(0))
        {
            candidate[ci.pointIndex] = r;
            minR = std::min(minR, r);
            maxR = std::max(maxR, r);
            sumR += r;
            ++validScaleCount;
        }
    }

    publishReferenceDelassus();

    return true;
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::commitLaggedComplianceSnapshot()
{
    if (!usesLaggedCompliance())
        return false;

    if (!ensureReferenceComplianceCache())
        return false;

    sofa::type::vector<Real> candidate;

    if (!computeLaggedComplianceFromCurrentContacts(candidate))
        return false;

    m_nextCompliance = std::move(candidate);
    m_hasNextCompliance = true;

    return true;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::ResidualBlockNorms
FischerBurmeisterContactForceField<T1, T2>::currentResidualBlockNorms() const
{
    ResidualBlockNorms result;

    if (this->mstate1)
    {
        const auto force = this->mstate1->read(core::vec_id::read_access::force);
        for (const Deriv1& value : force->getValue())
            for (sofa::Size d = 0; d < DerivDim1; ++d)
                result.mechanicalSquaredNorm += static_cast<SReal>(value[d]) * static_cast<SReal>(value[d]);
    }

    if (this->mstate2)
    {
        const auto force = this->mstate2->read(core::vec_id::read_access::force);
        for (const Deriv2& value : force->getValue())
            for (sofa::Size d = 0; d < DerivDim2; ++d)
                result.complementaritySquaredNorm += static_cast<SReal>(value[d]) * static_cast<SReal>(value[d]);
    }

    return result;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::ContactDiagnostics
FischerBurmeisterContactForceField<T1, T2>::summarizeContacts() const
{
    ContactDiagnostics out;
    Real minGap = std::numeric_limits<Real>::infinity();
    Real minLambda = std::numeric_limits<Real>::infinity();
    Real maxLambda = -std::numeric_limits<Real>::infinity();

    for (const Contact& c : m_contacts)
    {
        switch (c.status)
        {
        case ContactStatus::Active:
            ++out.activeCount;
            out.hasActiveContact = true;
            minGap = std::min(minGap, c.gap);
            break;
        case ContactStatus::Pinned:
            ++out.pinnedCount;
            break;
        case ContactStatus::InvalidGeometry:
            ++out.invalidCount;
            break;
        }

        minLambda = std::min(minLambda, c.lambda);
        maxLambda = std::max(maxLambda, c.lambda);
        out.phiSquaredNorm += c.phi * c.phi;
    }

    if (out.hasActiveContact)
    {
        out.minimumActiveGap = minGap;
        out.maximumPenetration = std::max(Real(0), -minGap);
    }

    if (!m_contacts.empty())
    {
        out.minimumLambda = minLambda;
        out.maximumLambda = maxLambda;
    }

    return out;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::ContactDiagnostics
FischerBurmeisterContactForceField<T1, T2>::currentContactDiagnostics() const
{
    return summarizeContacts();
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::storeFiniteDifferenceBase()
{
    m_fdBaseContacts.clear();

    if (!m_validState || !this->mstate1)
        return;

    const auto x1Data = this->mstate1->read(core::vec_id::read_access::position);
    const VecCoord1& x1 = x1Data->getValue();
    if (x1.size() != m_contacts.size())
        return;

    m_fdBaseContacts.resize(m_contacts.size());
    for (sofa::Index i = 0; i < m_contacts.size(); ++i)
    {
        m_fdBaseContacts[i].contact = m_contacts[i];
        m_fdBaseContacts[i].position = extractPosition(x1[i]);
    }
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::logFiniteDifferenceTrial(Real alpha) const
{
    if (!(alpha > Real(0)) || !this->mstate1 || m_fdBaseContacts.size() != m_contacts.size())
        return;

    const auto x1Data = this->mstate1->read(core::vec_id::read_access::position);
    const VecCoord1& x1 = x1Data->getValue();
    if (x1.size() != m_contacts.size())
        return;

    const Real eps = Real(1e-14);

    // Diagnostic thresholds. Keep these hard-coded while the diagnostics are
    // being tuned; they can become Data<> fields once the useful ranges settle.
    const Real normalAngleTolDeg = Real(2);
    const Real gradientChangeTol = Real(0.05);
    const Real dgAbsTol = Real(1e-6);
    const Real dgRelTol = Real(0.01);
    const Real dHAbsTol = Real(1e-5);
    const Real dHRelTol = Real(0.05);
    const Real dForceAbsTol = Real(1e-4);
    const Real dForceRelTol = Real(0.01);
    const Real dPhiAbsTol = Real(1e-6);
    const Real dPhiRelTol = Real(0.01);
    const Real complianceFreezeTol = Real(1e-12);

    auto scalarScore = [eps](Real fd, Real jac, Real absTol, Real relTol)
    {
        const Real error = std::abs(fd - jac);
        const Real scale = std::max(std::abs(fd), std::abs(jac));
        const Real allowed = std::max(absTol, relTol * scale);
        return error / std::max(allowed, eps);
    };

    auto vectorScore = [eps](const Vec3& fd, const Vec3& jac, Real absTol, Real relTol)
    {
        Vec3 error;
        error.clear();
        for (sofa::Size d = 0; d < TranslationalDim; ++d)
            error[d] = fd[d] - jac[d];

        const Real errorNorm = std::sqrt(error.norm2());
        const Real fdNorm = std::sqrt(fd.norm2());
        const Real jacNorm = std::sqrt(jac.norm2());
        const Real allowed = std::max(absTol, relTol * std::max(fdNorm, jacNorm));
        return errorNorm / std::max(allowed, eps);
    };

    auto relativeScalarError = [eps](Real fd, Real jac)
    {
        return std::abs(fd - jac) / std::max(std::max(std::abs(fd), std::abs(jac)), eps);
    };

    auto relativeVectorError = [eps](const Vec3& fd, const Vec3& jac)
    {
        Vec3 error;
        error.clear();
        for (sofa::Size d = 0; d < TranslationalDim; ++d)
            error[d] = fd[d] - jac[d];

        const Real errorNorm = std::sqrt(error.norm2());
        const Real fdNorm = std::sqrt(fd.norm2());
        const Real jacNorm = std::sqrt(jac.norm2());
        return errorNorm / std::max(std::max(fdNorm, jacNorm), eps);
    };

    struct WorstContact
    {
        bool valid = false;
        sofa::Index point = 0;
        std::string flags;
        Real score = Real(0);
        Real g = Real(0);
        Real lambda = Real(0);
        Real phi = Real(0);
        Real angleDeg = Real(0);
        Real dgRelErr = Real(0);
        Real dHRelErr = Real(0);
        Real dFRelErr = Real(0);
        Real dPhiRelErr = Real(0);
        Real dr = Real(0);
        Real gradNorm = Real(0);
        Real hessGradNorm = Real(0);
        Real gapHessFrob = Real(0);
        Real lambdaHessFrob = Real(0);
        bool hasHessian = false;
        bool dForceOutlier = false;
        Vec3 dForceFD;
        Vec3 dForceJ;
        Vec3 lambdaPart;
        Vec3 hessianPart;
        Vec3 dHfd;
        Vec3 dHj;
    };

    WorstContact worst;

    sofa::Size activeCount = 0;
    sofa::Size hessianCount = 0;
    sofa::Size outlierCount = 0;
    sofa::Size statusChangeCount = 0;

    bool hasFirstStatusChange = false;
    sofa::Index firstStatusPoint = 0;
    unsigned int firstBaseStatus = 0;
    unsigned int firstTrialStatus = 0;

    Real minimumGradientNorm = std::numeric_limits<Real>::infinity();
    Real maximumGradientNorm = Real(0);
    Real maximumGradientNormError = Real(0);
    sofa::Index maximumGradientNormErrorPoint = 0;

    Real maximumHessianFrob = Real(0);
    sofa::Index maximumHessianFrobPoint = 0;

    Real maximumHessianGradientNorm = Real(0);
    sofa::Index maximumHessianGradientPoint = 0;

    Real maximumEikonalRelativeDefect = Real(0);
    sofa::Index maximumEikonalRelativeDefectPoint = 0;

    Real maximumGapHessianFrob = Real(0);
    sofa::Index maximumGapHessianPoint = 0;

    Real maximumLambdaHessianFrob = Real(0);
    sofa::Index maximumLambdaHessianPoint = 0;

    for (sofa::Index i = 0; i < m_contacts.size(); ++i)
    {
        const auto& snapshot = m_fdBaseContacts[i];
        const Contact& base = snapshot.contact;
        const Contact& trial = m_contacts[i];

        if (base.status != trial.status)
        {
            ++statusChangeCount;
            if (!hasFirstStatusChange)
            {
                hasFirstStatusChange = true;
                firstStatusPoint = i;
                firstBaseStatus = static_cast<unsigned int>(base.status);
                firstTrialStatus = static_cast<unsigned int>(trial.status);
            }
            continue;
        }

        if (base.status != ContactStatus::Active)
            continue;

        ++activeCount;

        const Real H0Norm = std::sqrt(base.gapGradient.norm2());
        const Real H1Norm = std::sqrt(trial.gapGradient.norm2());
        minimumGradientNorm = std::min(minimumGradientNorm, H0Norm);
        maximumGradientNorm = std::max(maximumGradientNorm, H0Norm);

        const Real gradientNormError = std::abs(H0Norm - Real(1));
        if (gradientNormError > maximumGradientNormError)
        {
            maximumGradientNormError = gradientNormError;
            maximumGradientNormErrorPoint = i;
        }

        Vec3 deltaH;
        deltaH.clear();
        Real Hdot = Real(0);

        for (sofa::Size d = 0; d < TranslationalDim; ++d)
        {
            deltaH[d] = trial.gapGradient[d] - base.gapGradient[d];
            Hdot += base.gapGradient[d] * trial.gapGradient[d];
        }

        const Real deltaHNorm = std::sqrt(deltaH.norm2());
        const Real gradientRelativeChange = deltaHNorm / std::max(H0Norm, eps);

        Real normalCosine = Real(1);
        if (H0Norm > eps && H1Norm > eps)
        {
            normalCosine = Hdot / (H0Norm * H1Norm);
            normalCosine = std::max(Real(-1), std::min(Real(1), normalCosine));
        }

        const Real normalAngleDeg = std::acos(normalCosine) * Real(57.29577951308232);
        const Vec3 trialPosition = extractPosition(x1[i]);

        Vec3 dx, dHfd, dHj, forceBase, forceTrial, dForceFD, dForceJ;
        dx.clear();
        dHfd.clear();
        dHj.clear();
        forceBase.clear();
        forceTrial.clear();
        dForceFD.clear();
        dForceJ.clear();

        Real dgJ = Real(0);
        for (sofa::Size d = 0; d < TranslationalDim; ++d)
        {
            dx[d] = (trialPosition[d] - snapshot.position[d]) / alpha;
            dHfd[d] = (trial.gapGradient[d] - base.gapGradient[d]) / alpha;
            forceBase[d] = base.lambda * base.gapGradient[d];
            forceTrial[d] = trial.lambda * trial.gapGradient[d];
            dForceFD[d] = (forceTrial[d] - forceBase[d]) / alpha;
            dgJ += base.gapGradient[d] * dx[d];
        }

        const Real dgFD = (trial.gap - base.gap) / alpha;
        const Real dLambda = (trial.lambda - base.lambda) / alpha;
        const Real deltaCompliance = trial.complianceScale - base.complianceScale;
        const Real dr = deltaCompliance / alpha;

        Mat3 hessian;
        const bool hasHessian = computeGapHessian(snapshot.position, hessian);

        Real hessianFrob = Real(0);
        Real hessianGradientNorm = Real(0);
        Real gapHessianFrob = Real(0);
        Real lambdaHessianFrob = Real(0);

        if (hasHessian)
        {
            ++hessianCount;
            Real hessianFrob2 = Real(0);
            Vec3 hessianGradient;
            hessianGradient.clear();

            for (sofa::Size row = 0; row < TranslationalDim; ++row)
            {
                for (sofa::Size col = 0; col < TranslationalDim; ++col)
                {
                    const Real value = hessian(row, col);
                    hessianFrob2 += value * value;
                    dHj[row] += value * dx[col];
                    hessianGradient[row] += value * base.gapGradient[col];
                }
            }

            hessianFrob = std::sqrt(hessianFrob2);
            hessianGradientNorm = std::sqrt(hessianGradient.norm2());
            const Real eikonalRelativeDefect = hessianGradientNorm / std::max(hessianFrob * H0Norm, eps);
            gapHessianFrob = std::abs(base.gap) * hessianFrob;
            lambdaHessianFrob = std::abs(base.lambda) * hessianFrob;

            if (hessianFrob > maximumHessianFrob)
            {
                maximumHessianFrob = hessianFrob;
                maximumHessianFrobPoint = i;
            }

            if (hessianGradientNorm > maximumHessianGradientNorm)
            {
                maximumHessianGradientNorm = hessianGradientNorm;
                maximumHessianGradientPoint = i;
            }

            if (eikonalRelativeDefect > maximumEikonalRelativeDefect)
            {
                maximumEikonalRelativeDefect = eikonalRelativeDefect;
                maximumEikonalRelativeDefectPoint = i;
            }

            if (gapHessianFrob > maximumGapHessianFrob)
            {
                maximumGapHessianFrob = gapHessianFrob;
                maximumGapHessianPoint = i;
            }

            if (lambdaHessianFrob > maximumLambdaHessianFrob)
            {
                maximumLambdaHessianFrob = lambdaHessianFrob;
                maximumLambdaHessianPoint = i;
            }
        }

        const Real dPhiFD = (trial.phi - base.phi) / alpha;
        const Real dPhiJ = base.dPhiDgap * dgJ + base.dPhiDlambda * dLambda;

        Vec3 lambdaPart;
        Vec3 hessianPart;
        lambdaPart.clear();
        hessianPart.clear();

        for (sofa::Size d = 0; d < TranslationalDim; ++d)
        {
            lambdaPart[d] = base.gapGradient[d] * dLambda;
            hessianPart[d] = base.lambda * dHj[d];
            dForceJ[d] = lambdaPart[d] + hessianPart[d];
        }

        const Real hScore = std::max(
            normalAngleDeg / std::max(normalAngleTolDeg, eps),
            gradientRelativeChange / std::max(gradientChangeTol, eps));
        const Real dgScore = scalarScore(dgFD, dgJ, dgAbsTol, dgRelTol);
        const Real dHScore = hasHessian ? vectorScore(dHfd, dHj, dHAbsTol, dHRelTol) : Real(0);
        const Real dForceScore = vectorScore(dForceFD, dForceJ, dForceAbsTol, dForceRelTol);
        const Real dPhiScore = scalarScore(dPhiFD, dPhiJ, dPhiAbsTol, dPhiRelTol);
        const Real complianceScore = std::abs(deltaCompliance) / std::max(complianceFreezeTol, eps);

        const bool hOutlier = hScore > Real(1);
        const bool dgOutlier = dgScore > Real(1);
        const bool dHOutlier = hasHessian && dHScore > Real(1);
        const bool dForceOutlier = dForceScore > Real(1);
        const bool dPhiOutlier = dPhiScore > Real(1);
        const bool complianceChanged = complianceScore > Real(1);

        if (!(hOutlier || dgOutlier || dHOutlier || dForceOutlier || dPhiOutlier || complianceChanged))
            continue;

        ++outlierCount;

        const Real score = std::max({hScore, dgScore, dHScore, dForceScore, dPhiScore, complianceScore});
        if (worst.valid && score <= worst.score)
            continue;

        std::ostringstream flags;
        if (hOutlier) flags << "H|";
        if (dgOutlier) flags << "DG|";
        if (dHOutlier) flags << "DH|";
        if (dForceOutlier) flags << "DF|";
        if (dPhiOutlier) flags << "DPHI|";
        if (complianceChanged) flags << "R|";

        std::string flagString = flags.str();
        if (!flagString.empty())
            flagString.pop_back();

        worst.valid = true;
        worst.point = i;
        worst.flags = std::move(flagString);
        worst.score = score;
        worst.g = base.gap;
        worst.lambda = base.lambda;
        worst.phi = base.phi;
        worst.angleDeg = normalAngleDeg;
        worst.dgRelErr = relativeScalarError(dgFD, dgJ);
        worst.dHRelErr = hasHessian ? relativeVectorError(dHfd, dHj) : Real(0);
        worst.dFRelErr = relativeVectorError(dForceFD, dForceJ);
        worst.dPhiRelErr = relativeScalarError(dPhiFD, dPhiJ);
        worst.dr = dr;
        worst.gradNorm = H0Norm;
        worst.hessGradNorm = hessianGradientNorm;
        worst.gapHessFrob = gapHessianFrob;
        worst.lambdaHessFrob = lambdaHessianFrob;
        worst.hasHessian = hasHessian;
        worst.dForceOutlier = dForceOutlier;
        worst.dForceFD = dForceFD;
        worst.dForceJ = dForceJ;
        worst.lambdaPart = lambdaPart;
        worst.hessianPart = hessianPart;
        worst.dHfd = dHfd;
        worst.dHj = dHj;
    }

    if (activeCount > 0)
    {
        msg_warning() << "[NCP SDF QUALITY]"
                      << " alpha=" << alpha
                      << " active=" << activeCount
                      << " hessian=" << hessianCount
                      << " |grad|=[" << minimumGradientNorm << "," << maximumGradientNorm << "]"
                      << " maxGradErr=" << maximumGradientNormError << "@" << maximumGradientNormErrorPoint
                      << " max|Hess|F=" << maximumHessianFrob << "@" << maximumHessianFrobPoint
                      << " max|Hess*grad|=" << maximumHessianGradientNorm << "@" << maximumHessianGradientPoint
                      << " maxEikonalRel=" << maximumEikonalRelativeDefect << "@" << maximumEikonalRelativeDefectPoint
                      << " max|g*Hess|F=" << maximumGapHessianFrob << "@" << maximumGapHessianPoint
                      << " max|lambda*Hess|F=" << maximumLambdaHessianFrob << "@" << maximumLambdaHessianPoint;
    }

    if (statusChangeCount > 0)
    {
        msg_warning() << "[NCP FD CONTACT STATUS]"
                      << " alpha=" << alpha
                      << " count=" << statusChangeCount
                      << " firstPoint=" << firstStatusPoint
                      << " first=" << firstBaseStatus << "->" << firstTrialStatus;
    }

    if (!worst.valid)
        return;

    const Real decades = std::log10(std::max(worst.score, Real(1)));
    msg_warning() << "[NCP FD CONTACT WORST]"
                  << " outliers=" << outlierCount
                  << " p=" << worst.point
                  << " flags=" << worst.flags
                  << " score=" << worst.score
                  << " decades=" << decades
                  << " g=" << worst.g
                  << " lambda=" << worst.lambda
                  << " phi=" << worst.phi
                  << " |grad|=" << worst.gradNorm
                  << " angle=" << worst.angleDeg
                  << " dgRel=" << worst.dgRelErr
                  << " dHRel=" << worst.dHRelErr
                  << " dFRel=" << worst.dFRelErr
                  << " dPhiRel=" << worst.dPhiRelErr
                  << " dr=" << worst.dr
                  << " |Hess*grad|=" << worst.hessGradNorm
                  << " |g*Hess|F=" << worst.gapHessFrob
                  << " |lambda*Hess|F=" << worst.lambdaHessFrob
                  << " Hess=" << worst.hasHessian;

    if (worst.dForceOutlier)
    {
        msg_warning() << "[NCP FD CONTACT DF]"
                      << " p=" << worst.point
                      << " |dFfd|=" << std::sqrt(worst.dForceFD.norm2())
                      << " |dFJ|=" << std::sqrt(worst.dForceJ.norm2())
                      << " |H*dLambda|=" << std::sqrt(worst.lambdaPart.norm2())
                      << " |lambda*HessDx|=" << std::sqrt(worst.hessianPart.norm2())
                      << " |dHfd|=" << std::sqrt(worst.dHfd.norm2())
                      << " |HessDx|=" << std::sqrt(worst.dHj.norm2());
    }
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::publishDebugData()
{
    const ContactDiagnostics diagnostics = summarizeContacts();
    d_activeContactCount.setValue(diagnostics.activeCount);
    d_pinnedContactCount.setValue(diagnostics.pinnedCount);
    d_invalidContactCount.setValue(diagnostics.invalidCount);

    if (!d_publishContactData.getValue())
        return;

    const sofa::Size n = m_contacts.size();
    sofa::type::vector<unsigned int> status(n);
    sofa::type::vector<Vec3> gradient(n);
    sofa::type::vector<Real> gap(n), lambda(n), r(n), scaledLambda(n), phi(n), dPhiDgap(n), beta(n);

    for (sofa::Index i = 0; i < n; ++i)
    {
        const Contact& c = m_contacts[i];
        status[i] = static_cast<unsigned int>(c.status);
        gradient[i] = c.gapGradient;
        gap[i] = c.gap;
        lambda[i] = c.lambda;
        r[i] = c.complianceScale;
        scaledLambda[i] = c.scaledLambda;
        phi[i] = c.phi;
        dPhiDgap[i] = c.dPhiDgap;
        beta[i] = c.dPhiDlambda;
    }

    d_contactStatus.setValue(status);
    d_contactGapGradient.setValue(gradient);
    d_contactGap.setValue(gap);
    d_contactLambda.setValue(lambda);
    d_contactComplianceScale.setValue(r);
    d_contactScaledLambda.setValue(scaledLambda);
    d_contactPhi.setValue(phi);
    d_contactDPhiDgap.setValue(dPhiDgap);
    d_contactDPhiDlambda.setValue(beta);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::addForce(const sofa::core::MechanicalParams*,DataVecDeriv1& dataF1,DataVecDeriv2& dataF2,const DataVecCoord1& dataX1,const DataVecCoord2& dataX2,const DataVecDeriv1&,const DataVecDeriv2&)
{
    if (!rebuildCurrentContacts(dataX1.getValue(), dataX2.getValue()))
        return;

    VecDeriv1& f1 = *dataF1.beginEdit();
    VecDeriv2& f2 = *dataF2.beginEdit();

    for (const Contact& c : m_contacts)
    {
        if (c.status == ContactStatus::Active)
        {
            // Physical contact residual: R_x^c = H^T lambda.
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                f1[c.pointIndex][d] += c.lambda * c.gapGradient[d];
        }

        f2[c.lambdaIndex][0] += c.phi;
    }

    dataF1.endEdit();
    dataF2.endEdit();
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::addDForce(const sofa::core::MechanicalParams* mparams,DataVecDeriv1& dataDF1,DataVecDeriv2& dataDF2,const DataVecDeriv1& dataDX1,const DataVecDeriv2& dataDX2)
{
    if (!m_validState || !this->mstate1)
        return;

    const VecDeriv1& dx1 = dataDX1.getValue();
    const VecDeriv2& dx2 = dataDX2.getValue();
    VecDeriv1& df1 = *dataDF1.beginEdit();
    VecDeriv2& df2 = *dataDF2.beginEdit();
    const Real k = mparams->kFactor();

    // Needed only by the optional J_xx = lambda Hess(g) contribution.
    const auto x1Data = this->mstate1->read(core::vec_id::read_access::position);
    const VecCoord1& x1 = x1Data->getValue();

    for (const Contact& c : m_contacts)
    {
        const Real deltaLambda = dx2[c.lambdaIndex][0];

        Real deltaGap = Real(0);
        for (sofa::Size d = 0; d < TranslationalDim; ++d)
            deltaGap += c.gapGradient[d] * dx1[c.pointIndex][d];

        if (c.status == ContactStatus::Active)
        {
            // d(H^T lambda)/d(lambda) * deltaLambda = H^T deltaLambda.
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                df1[c.pointIndex][d] += k * c.gapGradient[d] * deltaLambda;

            // Optional d(H^T lambda)/dx * deltaX
            //          = lambda Hess(g) deltaX.
            Mat3 gapHessian;
            if (c.lambda != Real(0)
                && c.pointIndex < x1.size()
                && computeGapHessian(extractPosition(x1[c.pointIndex]), gapHessian))
            {
                for (sofa::Size row = 0; row < TranslationalDim; ++row)
                {
                    Real value = Real(0);
                    for (sofa::Size col = 0; col < TranslationalDim; ++col)
                        value += gapHessian(row, col) * dx1[c.pointIndex][col];

                    df1[c.pointIndex][row] += k * c.lambda * value;
                }
            }
        }

        // d phi = a H deltaX + b deltaLambda.
        df2[c.lambdaIndex][0] += k * (
            c.dPhiDgap * deltaGap
            + c.dPhiDlambda * deltaLambda);
    }

    dataDF1.endEdit();
    dataDF2.endEdit();
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::addKToMatrix(const sofa::core::MechanicalParams*,const sofa::core::behavior::MultiMatrixAccessor*)
{
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    if (!matrix || !m_validState || !this->mstate1 || !this->mstate2)
        return;

    // Exact contact Jacobian for
    //
    //     R_x      = H^T lambda
    //     R_lambda = phi(g, r lambda)
    //
    // with r frozen during this linearization:
    //
    //         [ lambda Hess(g)    H^T ]
    //     J = [                       ]
    //         [      a H          b   ]
    //
    // The Hessian hook returns false in the base class, so J_xx is zero until
    // a geometry specialization provides Hess(g).

    auto dRx_dX = matrix->getForceDerivativeIn(this->mstate1.get())
        .withRespectToPositionsIn(this->mstate1.get());
    auto dRx_dLambda = matrix->getForceDerivativeIn(this->mstate1.get())
        .withRespectToPositionsIn(this->mstate2.get());
    auto dPhi_dX = matrix->getForceDerivativeIn(this->mstate2.get())
        .withRespectToPositionsIn(this->mstate1.get());
    auto dPhi_dLambda = matrix->getForceDerivativeIn(this->mstate2.get())
        .withRespectToPositionsIn(this->mstate2.get());

    dRx_dX.checkValidity(this);
    dRx_dLambda.checkValidity(this);
    dPhi_dX.checkValidity(this);
    dPhi_dLambda.checkValidity(this);

    sofa::type::Mat<DerivDim1, DerivDim1, Real> upperLeft;
    sofa::type::Mat<DerivDim1, DerivDim2, Real> upperRight;
    sofa::type::Mat<DerivDim2, DerivDim1, Real> lowerLeft;
    sofa::type::Mat<DerivDim2, DerivDim2, Real> lowerRight;

    const auto x1Data = this->mstate1->read(core::vec_id::read_access::position);
    const VecCoord1& x1 = x1Data->getValue();

    for (const Contact& c : m_contacts)
    {
        upperLeft.clear();
        upperRight.clear();
        lowerLeft.clear();
        lowerRight.clear();

        if (c.status == ContactStatus::Active)
        {
            // J_xlambda = d(H^T lambda)/d(lambda) = H^T.
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                upperRight(d, 0) = c.gapGradient[d];

            // J_lambdax = dphi/dg * dg/dx = a H.
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                lowerLeft(0, d) = c.dPhiDgap * c.gapGradient[d];

            // Optional J_xx = lambda Hess(g).
            Mat3 gapHessian;
            if (c.lambda != Real(0) && c.pointIndex < x1.size() && computeGapHessian(extractPosition(x1[c.pointIndex]), gapHessian))
            {
                for (sofa::Size row = 0; row < TranslationalDim; ++row)
                    for (sofa::Size col = 0; col < TranslationalDim; ++col)
                        upperLeft(row, col) = c.lambda * gapHessian(row, col);
            }
        }

        // Active row: dphi/dlambda = b.
        // Pinned/invalid row: finalizeContactRow() sets b=1 and H=0, yielding the simple equation lambda=0.
        lowerRight(0, 0) = c.dPhiDlambda;

        if (c.status == ContactStatus::Active)
            lowerRight(0, 0) += d_contactNewtonRegularization.getValue();

        dRx_dX(
            DerivDim1 * c.pointIndex,
            DerivDim1 * c.pointIndex) += upperLeft;

        dRx_dLambda(
            DerivDim1 * c.pointIndex,
            DerivDim2 * c.lambdaIndex) += upperRight;

        dPhi_dX(
            DerivDim2 * c.lambdaIndex,
            DerivDim1 * c.pointIndex) += lowerLeft;

        dPhi_dLambda(
            DerivDim2 * c.lambdaIndex,
            DerivDim2 * c.lambdaIndex) += lowerRight;
    }
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
}

template<class T1, class T2>
SReal FischerBurmeisterContactForceField<T1, T2>::getPotentialEnergy(const sofa::core::MechanicalParams*,const DataVecCoord1&,const DataVecCoord2&) const
{
    return SReal(0);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::draw(const core::visual::VisualParams* vparams)
{
    drawActiveNormals(vparams);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::drawActiveNormals(const core::visual::VisualParams* vparams) const
{
    if (!vparams || !vparams->drawTool() || !d_showContactGradients.getValue() || !this->mstate1)
        return;

    const auto x1Data = this->mstate1->read(core::vec_id::read_access::position);
    const VecCoord1& x1 = x1Data->getValue();
    std::vector<sofa::type::Vec3> lines;
    lines.reserve(2 * m_contacts.size());
    const Real scale = d_drawGradientScale.getValue();

    for (const Contact& c : m_contacts)
    {
        const Real norm2 = c.gapGradient.norm2();
        if (c.status != ContactStatus::Active
            || c.pointIndex >= x1.size()
            || c.gap > Real(1e-2)
            || norm2 <= Real(1e-30))
        {
            continue;
        }

        const Vec3 p0 = extractPosition(x1[c.pointIndex]);
        const Vec3 p1 = p0 + c.gapGradient * (scale / std::sqrt(norm2));
        lines.emplace_back(static_cast<float>(p0[0]), static_cast<float>(p0[1]), static_cast<float>(p0[2]));
        lines.emplace_back(static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]));
    }

    if (!lines.empty())
        vparams->drawTool()->drawLines(lines, 2.0f, d_contactColor.getValue());
}

} // namespace sofa::ncp