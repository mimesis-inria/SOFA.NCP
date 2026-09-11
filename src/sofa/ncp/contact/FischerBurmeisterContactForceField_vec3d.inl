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
#include <sofa/linearalgebra/BaseMatrix.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace sofa::ncp
{

template<class T1, class T2>
FischerBurmeisterContactForceField<T1, T2>::FischerBurmeisterContactForceField()
    : Inherit()
    , d_fixedComplianceScale(initData(&d_fixedComplianceScale, Real(1.0),
          "fixedComplianceScale", "Positive fallback r used in phi(g,r*lambda)/r."))
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
    , l_contactMapping(initLink("contactMapping", "External Rigid3-to-Vec3 mapping; constant Jacobian assumed between reinit calls."))
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
    , d_contactGapGradient(initData(&d_contactGapGradient, "contactGapGradient", "Contact-position gradient N=dg/dy."))
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
          "Row-major W_ref = N J0 P K_ref,f^{-1} P^T J0^T N^T."))
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
    m_contacts.clear();
    m_fdBaseContacts.clear();
    m_validState = false;
    auto fail = [this](const char* reason) { msg_error() << reason; return false; };

    if (!this->mstate1 || !this->mstate2)
        return fail("Contact and multiplier states are required.");
    if (this->mstate1->getSize() != this->mstate2->getSize())
        return fail("Use one scalar multiplier per contact point, not per beam node.");
    if (!l_beamForceField || !l_beamForceField->getMState())
        return fail("beamForceField must reference the parent Rigid3 beam (also supplies radius).");
    if (d_complianceMode.getValue() > Lagged)
        return fail("Supported compliance modes: 0=fixed, 1=lagged; 2 remains reserved.");
    const Real r = d_fixedComplianceScale.getValue();
    const Real eps = d_fbEpsilon.getValue();
    const Real reg = d_contactNewtonRegularization.getValue();
    const Real radius = l_beamForceField->d_radius.getValue();
    if (!std::isfinite(r) || r <= 0 || !std::isfinite(eps) || eps < 0
        || !std::isfinite(reg) || reg < 0 || !std::isfinite(radius) || radius < 0)
        return fail("Require finite r>0, epsilon>=0, Newton regularization>=0 and radius>=0.");

    if constexpr (std::is_same_v<T1, sofa::defaulttype::Vec3Types>)
    {
        if (!l_contactMapping
            || l_contactMapping->getFromModel() != l_beamForceField->getMState()
            || l_contactMapping->getToModel() != this->mstate1.get())
            return fail("contactMapping must connect beamForceField's state to object1.");
    }
    else if constexpr (std::is_same_v<T1, sofa::defaulttype::Rigid3Types>)
    {
        if (l_contactMapping || this->mstate1.get() != l_beamForceField->getMState())
            return fail("Direct Rigid3 contacts must use the beam state and no contactMapping.");
    }
    else
        return fail("Supported contact states: direct Rigid3 or mapped Vec3.");

    if (usesLaggedCompliance() && !l_fixedConstraint)
        return fail("Lagged compliance requires the parent beam's FixedProjectiveConstraint.");
    if (l_fixedConstraint && l_fixedConstraint->getMState() != l_beamForceField->getMState())
        return fail("fixedConstraint must belong to the parent beam.");
    return true;
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::init()
{
    Inherit::init();
    m_validState = false;
    // Referenced components may be in child nodes whose init() has not run yet.
    // Allow the geometry specialization to initialize; bwdInit validates dependencies.
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Invalid)
        this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::bwdInit()
{
    Inherit::bwdInit();
    // Preserve an initialization failure reported by the base or geometry specialization.
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    const bool valid = initializeContactRows();
    this->d_componentState.setValue(valid ? core::objectmodel::ComponentState::Valid
                                         : core::objectmodel::ComponentState::Invalid);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::reinit()
{
    Inherit::reinit();
    invalidateReferenceComplianceCache();
    const bool valid = initializeContactRows();
    this->d_componentState.setValue(valid ? core::objectmodel::ComponentState::Valid
                                         : core::objectmodel::ComponentState::Invalid);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::usesLaggedCompliance() const
{
    return d_complianceMode.getValue() == Lagged;
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::usesCurrentCompliance() const
{
    return d_complianceMode.getValue() == Current;
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
    return std::isfinite(value) && value > Real(0) ? value : fallback;
}

template<class T1, class T2>
typename FischerBurmeisterContactForceField<T1, T2>::Vec3
FischerBurmeisterContactForceField<T1, T2>::extractPosition(const Coord1& coordinate)
{
    const auto p = DataTypes1::getCPos(coordinate);
    return Vec3(p[0], p[1], p[2]);
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
    return std::isfinite(c.gap)
        && std::isfinite(c.lambda)
        && std::isfinite(c.gapGradient[0])
        && std::isfinite(c.gapGradient[1])
        && std::isfinite(c.gapGradient[2])
        && std::isfinite(norm2)
        && norm2 > Real(1e-30);
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::updateFischerBurmeisterTerms(Contact& c) const
{
    const Real r = c.complianceScale;
    const Real eps = d_fbEpsilon.getValue();
    const Real g = c.gap;
    const Real s = r * c.lambda;
    const Real norm = std::hypot(std::hypot(g, s), eps);
    c.scaledLambda = s;

    if (norm == 0)
    {
        // Valid Clarke generalized derivative: approach along g = r*lambda > 0.
        const Real a = 1 - 1 / std::sqrt(Real(2));
        c.phi = 0;
        c.dPhiDgap = a / r;
        c.dPhiDlambda = a;
        return;
    }
    // Rationalize near cancellation, using normalized products to avoid squaring large inputs.
    const Real u = g / norm;
    const Real v = s / norm;
    const Real e = eps / norm;
    const Real denominator = u + v + 1;
    c.phi = denominator > Real(0.5)
        ? ((2 * u * s - e * eps) / denominator) / r
        : (g / r + c.lambda) - norm / r;
    c.dPhiDgap = (1 - u) / r;
    c.dPhiDlambda = 1 - v;
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
void FischerBurmeisterContactForceField<T1, T2>::finalizeContactRow(Contact& c, ContactStatus status, Real fixedR) const
{
    // Only direct beam contacts share indices with the beam constraint.
    if constexpr (std::is_same_v<T1, sofa::defaulttype::Rigid3Types>)
    {
        if (const auto* constraint = l_fixedConstraint.get())
        {
            const auto& indices = constraint->d_indices.getValue();
            if (constraint->fixAllDOFs() || std::find(indices.begin(), indices.end(), c.pointIndex) != indices.end())
                status = ContactStatus::Pinned;
        }
    }
    c.status = status;
    if (status == ContactStatus::Active && !hasValidKinematics(c))
        c.status = ContactStatus::InvalidGeometry;

    if (c.status != ContactStatus::Active)
    {
        c.gap = 0;
        c.gapGradient.clear();
        c.rotationalGapGradient.clear();
        c.complianceScale = 1;
        c.scaledLambda = c.lambda;
        c.phi = c.lambda;
        c.dPhiDgap = 0;
        c.dPhiDlambda = 1;
        return;
    }
    c.gap -= l_beamForceField->d_radius.getValue();
    c.complianceScale = complianceForPoint(c.pointIndex, fixedR);
    updateFischerBurmeisterTerms(c);
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::rebuildCurrentContacts(const VecCoord1& x1, const VecCoord2& x2)
{
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return false;

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
        c.lambda = x2[i][0];

        const ContactStatus status = computeContactKinematics(extractPosition(x1[i]), c);
        finalizeContactRow(c, status, fixedR);
        valid = valid && c.status != ContactStatus::InvalidGeometry && std::isfinite(c.phi)
            && std::isfinite(c.dPhiDgap) && std::isfinite(c.dPhiDlambda);
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
        indices.push_back(index);

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

    block = m_referenceTranslationalComplianceBlocks[pointI * m_referenceCompliancePointCount + pointJ];
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
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return false;
    if (!this->mstate1 || !l_beamForceField || !l_fixedConstraint)
        return false;
    const auto* beam = l_beamForceField.get();
    const auto* state = beam->getMState();
    if (!state)
        return false;
    const sofa::Size version = beam->getReferenceElasticMetricVersion();
    const std::size_t signature = fixedConstraintSignature();
    if (m_referenceComplianceCacheValid && m_cachedReferenceMetricVersion == version
        && m_cachedConstraintSignature == signature && m_cachedBeamState == state
        && m_cachedMapping == l_contactMapping.get() && m_cachedBeamPointCount == state->getSize()
        && m_referenceCompliancePointCount == this->mstate1->getSize())
        return true;
    return rebuildReferenceComplianceCache();
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::rebuildReferenceComplianceCache()
{
    invalidateReferenceComplianceCache();
    auto* beam = l_beamForceField.get();
    auto* constraint = l_fixedConstraint.get();
    if (!beam || !beam->getMState() || !constraint || !this->mstate1)
        return false;

    using Sparse = Eigen::SparseMatrix<Real>;
    using Dense = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    const sofa::Size nb = beam->getMState()->getSize();
    const sofa::Size nc = this->mstate1->getSize();
    if (nb == 0 || nc == 0)
        return false;

    std::vector<bool> fixed(nb, constraint->fixAllDOFs());
    for (const sofa::Index i : constraint->d_indices.getValue())
    {
        if (i >= nb)
        {
            msg_error() << "Fixed beam index out of bounds: " << i;
            return false;
        }
        fixed[i] = true;
    }
    std::vector<int> freeIndex(6 * nb, -1);
    int nf = 0;
    for (sofa::Index i = 0; i < nb; ++i)
        if (!fixed[i])
            for (int d = 0; d < 6; ++d)
                freeIndex[6 * i + d] = nf++;
    if (nf == 0)
    {
        msg_error() << "Reference compliance requires at least one free beam DOF.";
        return false;
    }

    std::vector<Eigen::Triplet<Real>> entries;
    const sofa::Size ne = beam->getReferenceElasticMetricElementCount();
    entries.reserve(144 * ne);
    typename BeamForceField::StiffnessMatrix Ke;
    for (sofa::Size e = 0; e < ne; ++e)
    {
        sofa::Index a, b;
        if (!beam->getReferenceElasticMetricElement(e, a, b, Ke) || a >= nb || b >= nb)
        {
            msg_error() << "Invalid reference beam element " << e;
            return false;
        }
        int reduced[12];
        for (int d = 0; d < 6; ++d)
        {
            reduced[d] = freeIndex[6 * a + d];
            reduced[6 + d] = freeIndex[6 * b + d];
        }
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
                if (reduced[i] >= 0 && reduced[j] >= 0)
                {
                    const Real value = (Ke(i,j) + Ke(j,i)) / 2;
                    if (value != 0)
                        entries.emplace_back(reduced[i], reduced[j], value);
                }
    }
    Sparse K(nf, nf);
    K.setFromTriplets(entries.begin(), entries.end());
    Eigen::SimplicialLDLT<Sparse> factor;
    factor.compute(K);
    if (factor.info() != Eigen::Success || !factor.vectorD().allFinite()
        || factor.vectorD().minCoeff() <= 0)
    {
        msg_error() << "Reduced reference beam metric is not positive definite.";
        return false;
    }

    // B = J0 P: contact translations versus ALL free beam DOFs, including rotations.
    Dense B = Dense::Zero(3 * nc, nf);
    if constexpr (std::is_same_v<T1, sofa::defaulttype::Vec3Types>)
    {
        // The external mapping must have evaluated its initial positions before this call.
        const auto* J = l_contactMapping->getJ();
        if (!J || J->rowSize() != 3 * nc || J->colSize() != 6 * nb)
        {
            msg_error() << "contactMapping Jacobian must have dimensions 3*Ncontacts by 6*Nbeam.";
            return false;
        }
        for (sofa::Size j = 0; j < 6 * nb; ++j)
            if (freeIndex[j] >= 0)
                for (sofa::Size i = 0; i < 3 * nc; ++i)
                    B(i, freeIndex[j]) = J->element(i, j);
    }
    else
    {
        if (nc != nb)
            return false;
        for (sofa::Index i = 0; i < nc; ++i)
            for (int d = 0; d < 3; ++d)
                if (freeIndex[6 * i + d] >= 0)
                    B(3 * i + d, freeIndex[6 * i + d]) = 1;
    }
    if (!B.allFinite())
    {
        msg_error() << "Nonfinite mapping Jacobian.";
        return false;
    }
    const Dense response = factor.solve(B.transpose());
    if (factor.info() != Eigen::Success || !response.allFinite())
    {
        msg_error() << "Reference compliance solve failed.";
        return false;
    }
    const Dense C = B * response;
    if (!C.allFinite())
        return false;
    m_referenceTranslationalComplianceBlocks.resize(nc * nc);
    for (sofa::Index i = 0; i < nc; ++i)
        for (sofa::Index j = 0; j < nc; ++j)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    m_referenceTranslationalComplianceBlocks[i * nc + j](r,c) =
                        (C(3*i+r, 3*j+c) + C(3*j+c, 3*i+r)) / 2;

    m_referenceCompliancePointCount = nc;
    m_cachedReferenceMetricVersion = beam->getReferenceElasticMetricVersion();
    m_cachedConstraintSignature = fixedConstraintSignature();
    m_cachedBeamState = beam->getMState();
    m_cachedBeamPointCount = nb;
    m_cachedMapping = l_contactMapping.get();
    m_referenceComplianceCacheValid = true;
    return true;
}

template<class T1, class T2>
bool FischerBurmeisterContactForceField<T1, T2>::computeLaggedComplianceFromCurrentContacts(sofa::type::vector<Real>& candidate)
{
    if (!m_validState || !m_referenceComplianceCacheValid)
        return false;
    const sofa::Size nc = m_referenceCompliancePointCount;
    candidate = m_hasCurrentCompliance ? m_currentCompliance
        : sofa::type::vector<Real>(nc, d_fixedComplianceScale.getValue());

    sofa::type::vector<const Contact*> active;
    m_referenceDelassusLambdaIndices.clear();
    for (const Contact& c : m_contacts)
        if (c.status == ContactStatus::Active)
        {
            active.push_back(&c);
            m_referenceDelassusLambdaIndices.push_back(c.lambdaIndex);
        }
    const sofa::Size n = active.size();
    m_referenceDelassus.assign(n * n, 0);
    for (sofa::Size i = 0; i < n; ++i)
    {
        const Contact& ci = *active[i];
        for (sofa::Size j = i; j < n; ++j)
        {
            const Contact& cj = *active[j];
            const Mat3& Cij = m_referenceTranslationalComplianceBlocks[ci.pointIndex * nc + cj.pointIndex];
            const Real w = ci.gapGradient * (Cij * cj.gapGradient);
            if (!std::isfinite(w))
            {
                clearReferenceDelassus();
                return false;
            }
            m_referenceDelassus[i * n + j] = m_referenceDelassus[j * n + i] = w;
        }
        // Zero projected normal mobility is not silently interpreted as a pinned contact.
        candidate[ci.pointIndex] = positiveOrFallback(m_referenceDelassus[i * n + i], candidate[ci.pointIndex]);
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
    // Read the parent residual AFTER child-force propagation and projective response.
    // Reading mapped child forces here would measure only contact forces.
    if (l_beamForceField && l_beamForceField->getMState())
    {
        const auto& force = l_beamForceField->getMState()->read(core::vec_id::read_access::force)->getValue();
        for (const auto& value : force)
            for (int d = 0; d < 6; ++d)
                result.mechanicalSquaredNorm += value[d] * value[d];
    }
    if (this->mstate2)
        for (const auto& value : this->mstate2->read(core::vec_id::read_access::force)->getValue())
            result.complementaritySquaredNorm += value[0] * value[0];
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
    if (!d_debug.getValue() || alpha <= 0 || !m_validState
        || m_fdBaseContacts.size() != m_contacts.size())
        return;
    const auto& x = this->mstate1->read(core::vec_id::read_access::position)->getValue();
    Real gapError = 0, phiError = 0, forceError = 0;
    sofa::Size skipped = 0;
    for (sofa::Index i = 0; i < m_contacts.size(); ++i)
    {
        const auto& base = m_fdBaseContacts[i];
        const Contact& a = base.contact;
        const Contact& b = m_contacts[i];
        if (a.status != b.status || a.complianceScale != b.complianceScale)
        {
            ++skipped;
            continue;
        }
        const Vec3 dy = (extractPosition(x[b.pointIndex]) - base.position) / alpha;
        const Real dl = (b.lambda - a.lambda) / alpha;
        const Real dg = a.gapGradient * dy;
        gapError = std::max(gapError, std::abs((b.gap - a.gap) / alpha - dg));
        phiError = std::max(phiError, std::abs((b.phi - a.phi) / alpha - a.dPhiDgap * dg - a.dPhiDlambda * dl));
        Vec3 predicted = a.gapGradient * dl;
        Mat3 hessian;
        if (a.status == ContactStatus::Active && computeGapHessian(base.position, hessian))
            predicted += (hessian * dy) * a.lambda;
        const Vec3 observed = (b.gapGradient * b.lambda - a.gapGradient * a.lambda) / alpha;
        forceError = std::max(forceError, (observed - predicted).norm());
    }
    // Local contact-space diagnostic; excludes intentional tangent regularization.
    msg_info() << "[NCP FD] alpha=" << alpha << " max|dg error|=" << gapError
               << " max|dphi error|=" << phiError << " max|df error|=" << forceError
               << " skipped=" << skipped;
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
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                df1[c.pointIndex][d] += k * c.gapGradient[d] * deltaLambda;

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

        df2[c.lambdaIndex][0] += k * (
            c.dPhiDgap * deltaGap
            + (c.dPhiDlambda + (c.status == ContactStatus::Active ? d_contactNewtonRegularization.getValue() : Real(0))) * deltaLambda);
    }

    dataDF1.endEdit();
    dataDF2.endEdit();
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::addKToMatrix(const sofa::core::MechanicalParams*,const sofa::core::behavior::MultiMatrixAccessor*)
{
    msg_error() << "Legacy addKToMatrix is unsupported; use buildStiffnessMatrix assembly.";
}

template<class T1, class T2>
void FischerBurmeisterContactForceField<T1, T2>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    if (!matrix || !m_validState || !this->mstate1 || !this->mstate2)
        return;

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
            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                upperRight(d, 0) = c.gapGradient[d];

            for (sofa::Size d = 0; d < TranslationalDim; ++d)
                lowerLeft(0, d) = c.dPhiDgap * c.gapGradient[d];

            Mat3 gapHessian;
            if (c.lambda != Real(0) && c.pointIndex < x1.size() && computeGapHessian(extractPosition(x1[c.pointIndex]), gapHessian))
            {
                for (sofa::Size row = 0; row < TranslationalDim; ++row)
                    for (sofa::Size col = 0; col < TranslationalDim; ++col)
                        upperLeft(row, col) = c.lambda * gapHessian(row, col);
            }
        }

        lowerRight(0, 0) = c.dPhiDlambda;

        if (c.status == ContactStatus::Active)
            lowerRight(0, 0) += d_contactNewtonRegularization.getValue();

        dRx_dX(DerivDim1 * c.pointIndex, DerivDim1 * c.pointIndex) += upperLeft;

        dRx_dLambda(DerivDim1 * c.pointIndex, DerivDim2 * c.lambdaIndex) += upperRight;

        dPhi_dX(DerivDim2 * c.lambdaIndex, DerivDim1 * c.pointIndex) += lowerLeft;

        dPhi_dLambda(DerivDim2 * c.lambdaIndex, DerivDim2 * c.lambdaIndex) += lowerRight;
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
            || c.lambda < Real(10.0)
            || norm2 <= Real(1e-30))
        {
            continue;
        }

        const Vec3 p0 = extractPosition(x1[c.pointIndex]);
        const Vec3 p1 = p0 + c.gapGradient * (scale / std::sqrt(norm2));
        lines.emplace_back(p0[0], p0[1], p0[2]);
        lines.emplace_back(p1[0], p1[1], p1[2]);
    }

    if (!lines.empty())
        vparams->drawTool()->drawLines(lines, 2.0f, d_contactColor.getValue());
}

} // namespace sofa::ncp