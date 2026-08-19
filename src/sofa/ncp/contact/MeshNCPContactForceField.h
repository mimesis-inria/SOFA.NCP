/****************************************************************************
* Triangle-mesh specialization of FischerBurmeisterContactForceField.
*
* Geometry queries are delegated to EnclosedCollisionPlugin's
* TriangleBVHCollisionModel, i.e. the same FCPW SAH/vectorized BVH used by the
* enclosing collision pipeline.
*
* The nonlinear contact law uses a local tangent-plane gap
*
*     g(x) = (x - q) . n - contactOffset,
*
* where q is the closest mesh point and n is the oriented triangle normal
* pointing toward the admissible lumen.
*
* This specialization intentionally supplies no gap Hessian. Consequently the
* contact residual still uses the exact queried gap/normal, while the Newton
* tangent is first-order in the contact geometry and contains no
* lambda*Hess(g) term.
****************************************************************************/
#pragma once

#include <sofa/ncp/config.h>
#include <sofa/ncp/contact/FischerBurmeisterContactForceField.h>

#include <EnclosedCollisionPlugin/TriangleBVHCollisionModel.h>

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>

#include <array>
#include <limits>
#include <vector>

namespace sofa::core { class ObjectFactory; }

namespace sofa::ncp
{

template<class TDataTypes1, class TDataTypes2>
class MeshNCPContactForceField final
    : public FischerBurmeisterContactForceField<TDataTypes1, TDataTypes2>
{
public:
    SOFA_CLASS(
        SOFA_TEMPLATE2(MeshNCPContactForceField, TDataTypes1, TDataTypes2),
        SOFA_TEMPLATE2(FischerBurmeisterContactForceField, TDataTypes1, TDataTypes2));

    using Inherit = FischerBurmeisterContactForceField<TDataTypes1, TDataTypes2>;
    using Real = typename Inherit::Real;
    using Vec3 = typename Inherit::Vec3;
    using Mat3 = typename Inherit::Mat3;
    using Contact = typename Inherit::Contact;
    using ContactStatus = typename Inherit::ContactStatus;
    using TriangleBVH = sofa::enclosedcollisionplugin::TriangleBVHCollisionModel;
    using TriangleState = sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>;
    using TriangleTopology = sofa::core::topology::BaseMeshTopology;

    SingleLink<MeshNCPContactForceField, TriangleBVH,
        BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_triangleCollisionModel;

    Data<bool> d_flipNormals;
    Data<Real> d_contactOffset;
    Data<Real> d_proximityThreshold;
    Data<bool> d_useBatchQueries;
    Data<bool> d_ignoreBoundaryEdges;
    Data<Real> d_boundaryBarycentricTolerance;
    Data<sofa::type::vector<unsigned int>> d_pinnedIndices;

    Data<sofa::Size> d_meshVertexCount;
    Data<sofa::Size> d_meshTriangleCount;
    Data<sofa::Size> d_meshBoundaryEdgeCount;

protected:
    MeshNCPContactForceField();

public:
    void init() override;
    void reinit() override;

protected:
    ContactStatus computeContactKinematics(const Vec3& position, Contact& contact) const override;
    bool computeGapHessian(const Vec3&, Mat3& hessian) const override;

private:
    static constexpr sofa::Index InvalidIndex = std::numeric_limits<sofa::Index>::max();

    TriangleState* m_triangleState { nullptr };
    TriangleTopology* m_triangleTopology { nullptr };
    sofa::type::vector<std::array<bool, 3>> m_boundaryEdges;

    mutable int m_cachedPositionCounter { -1 };
    mutable sofa::Size m_cachedPointCount { 0 };
    mutable std::vector<TriangleBVH::ClosestTriangleResult> m_closestResults;

    bool initializeMeshOracle();
    bool rebuildBoundaryCache();
    bool ensureClosestPointCache() const;
    bool queryClosestPoint(const Vec3& position, sofa::Index pointIndex,
        TriangleBVH::ClosestTriangleResult& result) const;
    bool triangleNormal(sofa::Index triangleIndex, Vec3& normal) const;
    bool liesOnIgnoredBoundary(sofa::Index triangleIndex, const Vec3& closestPoint) const;
    bool isPinned(sofa::Index pointIndex) const;

    static bool barycentricCoordinates(const Vec3& p, const Vec3& a, const Vec3& b,
        const Vec3& c, Vec3& barycentric);
    static Real dot(const Vec3& a, const Vec3& b);
    static Vec3 cross(const Vec3& a, const Vec3& b);
    static bool normalize(Vec3& value);
};

#if !defined(SOFANCP_MESH_NCP_CONTACT_FORCE_FIELD_CPP)
extern template class SOFANCP_API MeshNCPContactForceField<defaulttype::Rigid3Types, defaulttype::Vec1Types>;
extern template class SOFANCP_API MeshNCPContactForceField<defaulttype::Vec3Types, defaulttype::Vec1Types>;
#endif

SOFANCP_API void registerMeshNCPContactForceField(sofa::core::ObjectFactory* factory);

} // namespace sofa::ncp
