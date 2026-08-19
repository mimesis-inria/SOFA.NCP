/****************************************************************************
* FCPW-backed triangle-mesh Fischer-Burmeister contact implementation.
****************************************************************************/
#pragma once

#include <sofa/ncp/contact/MeshNCPContactForceField.h>
#include <sofa/ncp/contact/FischerBurmeisterContactForceField.inl>

#include <sofa/core/objectmodel/ComponentState.h>
#include <sofa/helper/logging/Messaging.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace sofa::ncp
{

template<class T1, class T2>
MeshNCPContactForceField<T1, T2>::MeshNCPContactForceField()
    : Inherit()
    , l_triangleCollisionModel(initLink(
          "triangleCollisionModel",
          "EnclosedCollisionPlugin TriangleBVHCollisionModel used for closest-point queries."))
    , d_flipNormals(initData(&d_flipNormals, false, "flipNormals",
          "Flip triangle normals. The final normal must point toward the admissible lumen."))
    , d_contactOffset(initData(&d_contactOffset, Real(0), "contactOffset",
          "Distance subtracted from the lumen-side gap, typically the catheter radius."))
    , d_proximityThreshold(initData(&d_proximityThreshold, Real(2), "proximityThreshold",
          "Only keep a valid mesh row active when its tangent-plane gap is <= this value. "
          "Negative disables proximity pinning. Penetrated rows are always active."))
    , d_useBatchQueries(initData(&d_useBatchQueries, true, "useBatchQueries",
          "Batch all object1 closest-point queries through FCPW once per position state."))
    , d_ignoreBoundaryEdges(initData(&d_ignoreBoundaryEdges, true, "ignoreBoundaryEdges",
          "Pin a row when its closest feature lies on an open mesh boundary edge."))
    , d_boundaryBarycentricTolerance(initData(&d_boundaryBarycentricTolerance, Real(1e-7),
          "boundaryBarycentricTolerance", "Tolerance used to detect a closest point on a boundary edge."))
    , d_pinnedIndices(initData(&d_pinnedIndices, sofa::type::vector<unsigned int>{ 0u },
          "pinnedIndices", "Object1 point indices excluded from mesh contact; index 0 is pinned by default."))
    , d_meshVertexCount(initData(&d_meshVertexCount, sofa::Size(0), "meshVertexCount",
          "Number of vertices in the linked vessel triangle model."))
    , d_meshTriangleCount(initData(&d_meshTriangleCount, sofa::Size(0), "meshTriangleCount",
          "Number of triangles in the linked vessel triangle model."))
    , d_meshBoundaryEdgeCount(initData(&d_meshBoundaryEdgeCount, sofa::Size(0), "meshBoundaryEdgeCount",
          "Number of open boundary edges in the linked vessel mesh."))
{
    d_meshVertexCount.setReadOnly(true);
    d_meshTriangleCount.setReadOnly(true);
    d_meshBoundaryEdgeCount.setReadOnly(true);
}

template<class T1, class T2>
void MeshNCPContactForceField<T1, T2>::init()
{
    Inherit::init();

    if (!initializeMeshOracle())
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
}

template<class T1, class T2>
void MeshNCPContactForceField<T1, T2>::reinit()
{
    Inherit::reinit();

    if (!initializeMeshOracle())
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::initializeMeshOracle()
{
    m_triangleState = nullptr;
    m_triangleTopology = nullptr;
    m_boundaryEdges.clear();
    m_closestResults.clear();
    m_cachedPositionCounter = -1;
    m_cachedPointCount = 0;

    d_meshVertexCount.setValue(0);
    d_meshTriangleCount.setValue(0);
    d_meshBoundaryEdgeCount.setValue(0);

    if (!l_triangleCollisionModel.get())
    {
        msg_error() << "triangleCollisionModel must link to an EnclosedCollisionPlugin::TriangleBVHCollisionModel.";
        return false;
    }

    if (!(d_contactOffset.getValue() >= Real(0))
        || !std::isfinite(static_cast<double>(d_contactOffset.getValue())))
    {
        msg_error() << "contactOffset must be finite and nonnegative.";
        return false;
    }

    if (!std::isfinite(static_cast<double>(d_proximityThreshold.getValue())))
    {
        msg_error() << "proximityThreshold must be finite.";
        return false;
    }

    if (!(d_boundaryBarycentricTolerance.getValue() >= Real(0))
        || !std::isfinite(static_cast<double>(d_boundaryBarycentricTolerance.getValue())))
    {
        msg_error() << "boundaryBarycentricTolerance must be finite and nonnegative.";
        return false;
    }

    auto* context = l_triangleCollisionModel->getContext();
    if (!context)
    {
        msg_error() << "triangleCollisionModel has no context.";
        return false;
    }

    m_triangleState = dynamic_cast<TriangleState*>(context->getMechanicalState());
    m_triangleTopology = context->getMeshTopology();

    if (!m_triangleState || !m_triangleTopology)
    {
        msg_error() << "triangleCollisionModel must share a Vec3 MechanicalState and triangle topology in its context.";
        return false;
    }

    l_triangleCollisionModel->computeBoundingTree(1);

    if (!rebuildBoundaryCache())
        return false;

    d_meshVertexCount.setValue(m_triangleState->getSize());
    d_meshTriangleCount.setValue(m_triangleTopology->getTriangles().size());

    msg_info() << "Mesh FB oracle ready: vertices=" << d_meshVertexCount.getValue()
               << " triangles=" << d_meshTriangleCount.getValue()
               << " boundaryEdges=" << d_meshBoundaryEdgeCount.getValue()
               << " batch=" << d_useBatchQueries.getValue()
               << " proximity=" << d_proximityThreshold.getValue()
               << " Hessian=disabled.";

    return true;
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::rebuildBoundaryCache()
{
    if (!m_triangleTopology)
        return false;

    const auto& triangles = m_triangleTopology->getTriangles();
    m_boundaryEdges.assign(triangles.size(), { false, false, false });

    using Edge = std::pair<sofa::Index, sofa::Index>;
    std::map<Edge, unsigned int> edgeCount;

    auto orderedEdge = [](sofa::Index a, sofa::Index b)
    {
        return a < b ? Edge(a, b) : Edge(b, a);
    };

    for (const auto& triangle : triangles)
    {
        ++edgeCount[orderedEdge(triangle[1], triangle[2])];
        ++edgeCount[orderedEdge(triangle[2], triangle[0])];
        ++edgeCount[orderedEdge(triangle[0], triangle[1])];
    }

    sofa::Size boundaryCount = 0;
    for (const auto& [edge, count] : edgeCount)
    {
        SOFA_UNUSED(edge);
        if (count == 1)
            ++boundaryCount;
    }

    for (sofa::Index i = 0; i < triangles.size(); ++i)
    {
        const auto& triangle = triangles[i];
        m_boundaryEdges[i][0] = edgeCount[orderedEdge(triangle[1], triangle[2])] == 1;
        m_boundaryEdges[i][1] = edgeCount[orderedEdge(triangle[2], triangle[0])] == 1;
        m_boundaryEdges[i][2] = edgeCount[orderedEdge(triangle[0], triangle[1])] == 1;
    }

    d_meshBoundaryEdgeCount.setValue(boundaryCount);
    return !triangles.empty();
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::ensureClosestPointCache() const
{
    if (!d_useBatchQueries.getValue() || !this->mstate1 || !l_triangleCollisionModel.get())
        return false;

    const auto xData = this->mstate1->read(core::vec_id::read_access::position);
    const int positionCounter = xData->getCounter();
    const auto& x = xData->getValue();

    if (m_cachedPositionCounter == positionCounter
        && m_cachedPointCount == x.size()
        && m_closestResults.size() == x.size())
    {
        return true;
    }

    l_triangleCollisionModel->computeBoundingTree(1);

    std::vector<fcpw::BoundingSphere<3>> queries;
    queries.reserve(x.size());

    for (const auto& coordinate : x)
    {
        const Vec3 p = Inherit::extractPosition(coordinate);
        queries.emplace_back(
            fcpw::Vector<3>(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])),
            fcpw::maxFloat);
    }

    l_triangleCollisionModel->getClosestTriangles(queries, m_closestResults);

    m_cachedPositionCounter = positionCounter;
    m_cachedPointCount = x.size();
    return m_closestResults.size() == x.size();
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::queryClosestPoint(
    const Vec3& position,
    sofa::Index pointIndex,
    TriangleBVH::ClosestTriangleResult& result) const
{
    if (d_useBatchQueries.getValue() && ensureClosestPointCache())
    {
        if (pointIndex >= m_closestResults.size())
            return false;

        result = m_closestResults[pointIndex];
        return result.triangleId != sofa::InvalidID;
    }

    l_triangleCollisionModel->computeBoundingTree(1);
    return l_triangleCollisionModel->getClosestTriangle(
        sofa::type::Vec3(position[0], position[1], position[2]), result);
}

template<class T1, class T2>
typename MeshNCPContactForceField<T1, T2>::ContactStatus
MeshNCPContactForceField<T1, T2>::computeContactKinematics(const Vec3& position, Contact& contact) const
{
    if (isPinned(contact.pointIndex))
        return ContactStatus::Pinned;

    if (!m_triangleState || !m_triangleTopology || !l_triangleCollisionModel.get())
        return ContactStatus::InvalidGeometry;

    TriangleBVH::ClosestTriangleResult result;
    if (!queryClosestPoint(position, contact.pointIndex, result))
        return ContactStatus::InvalidGeometry;

    if (result.triangleId == sofa::InvalidID
        || result.triangleId >= m_triangleTopology->getTriangles().size())
    {
        return ContactStatus::InvalidGeometry;
    }

    const Vec3 q(
        static_cast<Real>(result.closestPosition[0]),
        static_cast<Real>(result.closestPosition[1]),
        static_cast<Real>(result.closestPosition[2]));

    if (liesOnIgnoredBoundary(result.triangleId, q))
        return ContactStatus::Pinned;

    Vec3 normal;
    if (!triangleNormal(result.triangleId, normal))
        return ContactStatus::InvalidGeometry;

    if (d_flipNormals.getValue())
        normal = -normal;

    const Real gap = dot(position - q, normal) - d_contactOffset.getValue();
    if (!std::isfinite(static_cast<double>(gap)))
        return ContactStatus::InvalidGeometry;

    const Real proximity = d_proximityThreshold.getValue();
    if (proximity >= Real(0) && gap > proximity)
        return ContactStatus::Pinned;

    contact.gap = gap;
    contact.gapGradient = normal;
    return ContactStatus::Active;
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::computeGapHessian(const Vec3&, Mat3& hessian) const
{
    // Deliberate local-plane quasi-Newton geometry. The mesh residual uses the
    // freshly queried closest point and normal, but the tangent does not attempt
    // to differentiate closest-feature/triangle switching.
    hessian.clear();
    return false;
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::triangleNormal(sofa::Index triangleIndex, Vec3& normal) const
{
    normal.clear();

    if (!m_triangleState || !m_triangleTopology)
        return false;

    const auto& triangles = m_triangleTopology->getTriangles();
    if (triangleIndex >= triangles.size())
        return false;

    const auto xData = m_triangleState->read(core::vec_id::read_access::position);
    const auto& x = xData->getValue();
    const auto& triangle = triangles[triangleIndex];

    if (triangle[0] >= x.size() || triangle[1] >= x.size() || triangle[2] >= x.size())
        return false;

    const Vec3 a(static_cast<Real>(x[triangle[0]][0]), static_cast<Real>(x[triangle[0]][1]), static_cast<Real>(x[triangle[0]][2]));
    const Vec3 b(static_cast<Real>(x[triangle[1]][0]), static_cast<Real>(x[triangle[1]][1]), static_cast<Real>(x[triangle[1]][2]));
    const Vec3 c(static_cast<Real>(x[triangle[2]][0]), static_cast<Real>(x[triangle[2]][1]), static_cast<Real>(x[triangle[2]][2]));

    normal = cross(b - a, c - a);
    return normalize(normal);
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::liesOnIgnoredBoundary(
    sofa::Index triangleIndex, const Vec3& closestPoint) const
{
    if (!d_ignoreBoundaryEdges.getValue())
        return false;

    if (!m_triangleState || !m_triangleTopology || triangleIndex >= m_boundaryEdges.size())
        return false;

    const auto& triangles = m_triangleTopology->getTriangles();
    if (triangleIndex >= triangles.size())
        return false;

    const auto xData = m_triangleState->read(core::vec_id::read_access::position);
    const auto& x = xData->getValue();
    const auto& triangle = triangles[triangleIndex];

    const Vec3 a(static_cast<Real>(x[triangle[0]][0]), static_cast<Real>(x[triangle[0]][1]), static_cast<Real>(x[triangle[0]][2]));
    const Vec3 b(static_cast<Real>(x[triangle[1]][0]), static_cast<Real>(x[triangle[1]][1]), static_cast<Real>(x[triangle[1]][2]));
    const Vec3 c(static_cast<Real>(x[triangle[2]][0]), static_cast<Real>(x[triangle[2]][1]), static_cast<Real>(x[triangle[2]][2]));

    Vec3 barycentric;
    if (!barycentricCoordinates(closestPoint, a, b, c, barycentric))
        return false;

    const Real tolerance = d_boundaryBarycentricTolerance.getValue();
    const auto& boundary = m_boundaryEdges[triangleIndex];

    // barycentric[i]==0 means the point lies on the edge opposite vertex i.
    return (boundary[0] && barycentric[0] <= tolerance)
        || (boundary[1] && barycentric[1] <= tolerance)
        || (boundary[2] && barycentric[2] <= tolerance);
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::isPinned(sofa::Index pointIndex) const
{
    const auto& pinned = d_pinnedIndices.getValue();
    return std::find(pinned.begin(), pinned.end(), static_cast<unsigned int>(pointIndex)) != pinned.end();
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::barycentricCoordinates(
    const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, Vec3& barycentric)
{
    const Vec3 v0 = b - a;
    const Vec3 v1 = c - a;
    const Vec3 v2 = p - a;
    const Real d00 = dot(v0, v0);
    const Real d01 = dot(v0, v1);
    const Real d11 = dot(v1, v1);
    const Real d20 = dot(v2, v0);
    const Real d21 = dot(v2, v1);
    const Real denominator = d00 * d11 - d01 * d01;

    if (!(std::abs(denominator) > Real(1e-30)))
        return false;

    const Real inv = Real(1) / denominator;
    barycentric[1] = (d11 * d20 - d01 * d21) * inv;
    barycentric[2] = (d00 * d21 - d01 * d20) * inv;
    barycentric[0] = Real(1) - barycentric[1] - barycentric[2];
    return true;
}

template<class T1, class T2>
typename MeshNCPContactForceField<T1, T2>::Real
MeshNCPContactForceField<T1, T2>::dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<class T1, class T2>
typename MeshNCPContactForceField<T1, T2>::Vec3
MeshNCPContactForceField<T1, T2>::cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]);
}

template<class T1, class T2>
bool MeshNCPContactForceField<T1, T2>::normalize(Vec3& value)
{
    const Real norm2 = value.norm2();
    if (!(norm2 > Real(1e-30)) || !std::isfinite(static_cast<double>(norm2)))
        return false;

    value /= std::sqrt(norm2);
    return true;
}

} // namespace sofa::ncp
