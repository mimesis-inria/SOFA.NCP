/****************************************************************************
* Contact laws for scalar normal constraints.
*
* This header intentionally has no SOFA dependencies.  Geometry evaluation,
* MechanicalState coupling, and matrix assembly remain in the force field.
*
* Unilateral:  gap >= 0, lambda >= 0, gap * lambda = 0.
* Bilateral:   gap = 0, lambda unrestricted.
****************************************************************************/
#pragma once

#include <cmath>

namespace sofa::ncp
{

enum class ContactLawMode : unsigned int
{
    Unilateral = 0,
    Bilateral = 1
};

template<class Real>
struct ContactLawEvaluation
{
    Real residual = Real(0);
    Real dResidualDGap = Real(0);
    Real dResidualDLambda = Real(0);
};

/**
 * Normalized Fischer-Burmeister equation:
 *
 *    R = [g + r*lambda - sqrt(g*g + (r*lambda)^2 + epsilon^2)] / r.
 *
 * For epsilon=0 and positive r, R=0 is equivalent to the Signorini
 * complementarity conditions.  The positive scale r is treated as frozen
 * when computing derivatives (as in the existing compliance machinery).
 */
template<class Real>
struct UnilateralContactLaw
{
    static ContactLawEvaluation<Real> evaluate(
        Real gap, Real lambda, Real scale, Real epsilon)
    {
        const Real scaledLambda = scale * lambda;
        // Nested hypot prevents overflow in the norm evaluation.
        const Real norm = std::hypot(std::hypot(gap, scaledLambda), epsilon);

        // At (gap, scaledLambda, epsilon)=(0,0,0), choose the symmetric
        // semismooth generalized derivative (u,v)=(1/sqrt(2),1/sqrt(2)).
        // This avoids 0/0 while retaining a valid generalized Jacobian.
        const Real u = norm > Real(0)
            ? gap / norm : Real(1) / std::sqrt(Real(2));
        const Real v = norm > Real(0)
            ? scaledLambda / norm : Real(1) / std::sqrt(Real(2));

        return {
            (gap + scaledLambda - norm) / scale,
            (Real(1) - u) / scale,
            Real(1) - v
        };
    }
};

/** Equality constraint: R = gap, with lambda free to have either sign. */
template<class Real>
struct BilateralContactLaw
{
    static ContactLawEvaluation<Real> evaluate(Real gap,  Real epsilon)
    {
        return {gap - epsilon, Real(1), Real(0)};
    }
};

} // namespace sofa::ncp
