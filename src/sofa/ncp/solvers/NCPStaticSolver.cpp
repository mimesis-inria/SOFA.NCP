#include <sofa/ncp/solvers/NCPStaticSolver.h>
#include <sofa/ncp/solvers/NCPDebugNewtonRaphsonSolver.h>
#include <sofa/ncp/solvers/NCPMechanicalContinuationFunctionInterface.h>
#include <sofa/ncp/solvers/NCPMechanicalContinuationNewtonRaphsonSolver.h>
#include <sofa/ncp/contact/FischerBurmeisterContactForceField.inl>

#include <sofa/component/odesolver/backward/NonLinearFunction.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sofa::ncp
{

void registerNCPStaticSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData(
        "Static Fischer-Burmeister backend with exact trial residuals and merit backtracking.")
        .add<NCPStaticSolver>());
}

NCPStaticSolver::NCPStaticSolver()
    : Inherit()
    , l_contactForceField(initLink("contactForceField", "Fischer-Burmeister contact force field."))
    , d_mechanicalResidualReference(initData(&d_mechanicalResidualReference, 1_sreal,
        "mechanicalResidualReference", "Positive mechanical residual scale."))
    , d_complementarityResidualReference(initData(&d_complementarityResidualReference, 1_sreal,
        "complementarityResidualReference", "Positive complementarity residual scale."))
    , d_debug(initData(&d_debug, false, "debug", "Print residual-evaluation summaries."))
    , d_finiteDifferenceCheck(initData(&d_finiteDifferenceCheck, false,
        "finiteDifferenceCheck", "Compare actual line-search residual derivatives against J*d and contact derivatives."))
    , d_adaptiveContactRegularization(initData(&d_adaptiveContactRegularization, false,
        "adaptiveContactRegularization",
        "Adaptively increase the tangent-only active-contact diagonal when the current contact Schur block is near singular."))
    , d_contactSchurSingularValueFloor(initData(&d_contactSchurSingularValueFloor, 1e-3_sreal,
        "contactSchurSingularValueFloor",
        "Minimum requested singular value of the active contact Schur block before Newton solve."))
    , d_contactSchurTargetConditionNumber(initData(&d_contactSchurTargetConditionNumber, 1e4_sreal,
        "contactSchurTargetConditionNumber",
        "Maximum requested active-contact Schur condition number; <=1 disables this criterion."))
    , d_contactSchurMaxAdditionalRegularization(initData(&d_contactSchurMaxAdditionalRegularization, 1_sreal,
        "contactSchurMaxAdditionalRegularization",
        "Maximum additional tangent-only contact diagonal added adaptively in one Newton assembly."))
    , d_logAdaptiveContactRegularization(initData(&d_logAdaptiveContactRegularization, false,
        "logAdaptiveContactRegularization",
        "Print contact Schur spectrum and adaptive regularization decisions."))
    , d_lastMechanicalResidualNorm(initData(&d_lastMechanicalResidualNorm, 0_sreal,
        "lastMechanicalResidualNorm", "Mechanical residual norm from the retained state."))
    , d_lastComplementarityResidualNorm(initData(&d_lastComplementarityResidualNorm, 0_sreal,
        "lastComplementarityResidualNorm", "Complementarity residual norm from the retained state."))
    , d_lastComplianceCaptureSucceeded(initData(&d_lastComplianceCaptureSucceeded, false,
        "lastComplianceCaptureSucceeded", "Whether compliance was captured after the last converged solve."))
    , d_lastContactSchurSigmaMinBefore(initData(&d_lastContactSchurSigmaMinBefore, 0_sreal,
        "lastContactSchurSigmaMinBefore", "Smallest active-contact Schur singular value before adaptive damping."))
    , d_lastContactSchurSigmaMinAfter(initData(&d_lastContactSchurSigmaMinAfter, 0_sreal,
        "lastContactSchurSigmaMinAfter", "Smallest active-contact Schur singular value after adaptive damping."))
    , d_lastContactSchurConditionBefore(initData(&d_lastContactSchurConditionBefore, 0_sreal,
        "lastContactSchurConditionBefore", "Active-contact Schur condition estimate before adaptive damping."))
    , d_lastAdaptiveContactRegularization(initData(&d_lastAdaptiveContactRegularization, 0_sreal,
        "lastAdaptiveContactRegularization", "Additional contact diagonal applied to the last assembled Newton matrix."))
{
    d_lastMechanicalResidualNorm.setReadOnly(true);
    d_lastComplementarityResidualNorm.setReadOnly(true);
    d_lastComplianceCaptureSucceeded.setReadOnly(true);
    d_lastContactSchurSigmaMinBefore.setReadOnly(true);
    d_lastContactSchurSigmaMinAfter.setReadOnly(true);
    d_lastContactSchurConditionBefore.setReadOnly(true);
    d_lastAdaptiveContactRegularization.setReadOnly(true);
}

void NCPStaticSolver::init()
{
    Inherit::init();

    if (!l_contactForceField)
    {
        l_contactForceField.set(getContext()->get<ContactForceField>(getContext()->getTags(),core::objectmodel::BaseContext::SearchDown));
    }

    auto* contact = l_contactForceField.get();
    auto* diagnosticNewton = dynamic_cast<NCPDebugNewtonRaphsonSolver*>(l_newtonSolver.get());

    if (!contact || !l_linearSolver.get() || !diagnosticNewton)
    {
        msg_error() << "NCPStaticSolver requires contactForceField, linearSolver and NCPDebugNewtonRaphsonSolver links.";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (!(d_mechanicalResidualReference.getValue() > 0_sreal)
        || !(d_complementarityResidualReference.getValue() > 0_sreal))
    {
        msg_error() << "mechanicalResidualReference and complementarityResidualReference must be positive.";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (!(d_contactSchurSingularValueFloor.getValue() > 0_sreal)
        || !(d_contactSchurMaxAdditionalRegularization.getValue() >= 0_sreal)
        || !std::isfinite(static_cast<double>(d_contactSchurSingularValueFloor.getValue()))
        || !std::isfinite(static_cast<double>(d_contactSchurTargetConditionNumber.getValue()))
        || !std::isfinite(static_cast<double>(d_contactSchurMaxAdditionalRegularization.getValue())))
    {
        msg_error() << "Adaptive contact regularization parameters must be finite; "
                    << "contactSchurSingularValueFloor must be positive and "
                    << "contactSchurMaxAdditionalRegularization must be nonnegative.";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

    d_componentState.setValue(core::objectmodel::ComponentState::Valid);
}

namespace
{

using BaseNonLinearFunction = sofa::component::odesolver::backward::newton_raphson::BaseNonLinearFunction;

class NCPStaticResidualFunction final
    : public BaseNonLinearFunction
    , public NCPDebugNewtonFunctionInterface
    , public NCPMechanicalContinuationFunctionInterface
{
public:
    using MechanicalOperations = sofa::simulation::common::MechanicalOperations;
    using ContactForceField = NCPStaticSolver::ContactForceField;

    NCPStaticResidualFunction(
        NCPStaticSolver& owner,
        MechanicalOperations& mechanicalOperations,
        core::behavior::MultiVecCoord& position,
        core::behavior::MultiVecDeriv& residual,
        core::behavior::MultiVecDeriv& correction,
        core::behavior::LinearSolver* linearSolver,
        ContactForceField* contact,
        core::MultiVecCoordId solveStateId,
        core::MultiVecCoordId newtonStateId,
        core::MultiVecCoordId continuationStateId,
        core::MultiVecCoordId mechanicalStateId,
        SReal mechanicalReference,
        SReal complementarityReference,
        bool debug,
        bool finiteDifferenceCheck)
        : m_owner(owner)
        , m_mechanicalOperations(mechanicalOperations)
        , m_position(position)
        , m_residual(residual)
        , m_correction(correction)
        , m_linearSolver(linearSolver)
        , m_contact(contact)
        , m_solveState(position.ops(), solveStateId)
        , m_newtonState(position.ops(), newtonStateId)
        , m_continuationState(position.ops(), continuationStateId)
        , m_mechanicalState(position.ops(), mechanicalStateId)
        , m_mechanicalReference(mechanicalReference)
        , m_complementarityReference(complementarityReference)
        , m_debug(debug)
        , m_finiteDifferenceCheck(finiteDifferenceCheck)
    {
    }

    /** Enforce the current prescribed positions before the solve-entry checkpoint. */
    void projectCurrentState()
    {
        projectAndPropagate();
    }

    /** Evaluate the complete true residual at the current x and lambda. */
    void evaluateCurrentGuess() override
    {
        SCOPED_TIMER("NCPResidualEvaluation");
        m_mechanicalOperations.computeForce(m_residual, true, true);
        m_mechanicalOperations.projectResponse(m_residual);

        m_summary = NCPDebugResidualSummary{};

        if (!m_contact->isCurrentEvaluationValid())
        {
            m_scaledResidualSquaredNorm = std::numeric_limits<SReal>::infinity();
            m_summary.valid = false;
            return;
        }

        const auto blocks = m_contact->currentResidualBlockNorms();
        const auto diagnostics = m_contact->currentContactDiagnostics();
        const SReal rawSquaredNorm = m_residual.dot(m_residual);

        m_complementarityResidualSquaredNorm = blocks.complementaritySquaredNorm;
        m_mechanicalResidualSquaredNorm = std::max(rawSquaredNorm - m_complementarityResidualSquaredNorm,SReal(0));

        m_scaledResidualSquaredNorm =
            m_mechanicalResidualSquaredNorm / (m_mechanicalReference * m_mechanicalReference)
            + m_complementarityResidualSquaredNorm / (m_complementarityReference * m_complementarityReference);

        m_summary.mechanicalResidualNorm = std::sqrt(m_mechanicalResidualSquaredNorm);
        m_summary.complementarityResidualNorm = std::sqrt(m_complementarityResidualSquaredNorm);
        m_summary.scaledMechanicalSquaredNorm = m_mechanicalResidualSquaredNorm / (m_mechanicalReference * m_mechanicalReference);
        m_summary.scaledComplementaritySquaredNorm = m_complementarityResidualSquaredNorm / (m_complementarityReference * m_complementarityReference);
        m_summary.minimumActiveGap = diagnostics.minimumActiveGap;
        m_summary.maximumPenetration = diagnostics.maximumPenetration;
        m_summary.minimumLambda = diagnostics.minimumLambda;
        m_summary.maximumLambda = diagnostics.maximumLambda;
        m_summary.activeContacts = diagnostics.activeCount;
        m_summary.pinnedContacts = diagnostics.pinnedCount;
        m_summary.invalidContacts = diagnostics.invalidCount;
        m_summary.valid = diagnostics.invalidCount == 0 && std::isfinite(static_cast<double>(m_scaledResidualSquaredNorm));

    }

    SReal squaredNormLastEvaluation() override
    {
        return m_scaledResidualSquaredNorm;
    }

    /** Assemble A=-dR/dz at the current accepted Newton base. */
    void computeGradientFromCurrentGuess() override
    {
        assembleLinearizedSystem();
    }

    /** Solve the current mixed linear system and dispatch dz to all states. */
    void solveLinearEquation() override
    {
        SCOPED_TIMER("NCPLinearSolve");
        auto* system = m_linearSolver->getLinearSystem();
        system->setSystemSolution(m_correction);
        system->setRHS(m_residual);
        m_linearSolver->solveSystem();
        system->dispatchSystemSolution(m_correction);
    }

    // void solveLinearEquation() override
    // {
    //     SCOPED_TIMER("NCPLMSolve");

    //     auto* system = m_linearSolver->getLinearSystem();
    //     system->setSystemSolution(m_correction);
    //     system->setRHS(m_residual);

    //     auto* matrix = system->getSystemBaseMatrix();
    //     auto* rhs = system->getSystemRHSBaseVector();
    //     auto* solution = system->getSystemSolutionBaseVector();

    //     if (!matrix || !rhs || !solution)
    //         return;

    //     const Eigen::Index rows = matrix->rowSize();
    //     const Eigen::Index cols = matrix->colSize();

    //     Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(rows + cols, cols);
    //     Eigen::VectorXd augmentedRhs = Eigen::VectorXd::Zero(rows + cols);

    //     for (Eigen::Index row = 0; row < rows; ++row)
    //     {
    //         augmentedRhs[row] = rhs->element(row);

    //         for (Eigen::Index col = 0; col < cols; ++col)
    //             augmented(row, col) = matrix->element(row, col);
    //     }

    //     const Eigen::Index lambdaDofs = static_cast<Eigen::Index>(lambdaDofCount());
    //     const Eigen::Index lambdaBegin = cols - lambdaDofs;

    //     const SReal muMechanical = 0.0;
    //     const SReal muLambda = 1e-4;

    //     for (Eigen::Index col = 0; col < cols; ++col)
    //     {
    //         const SReal mu = col < lambdaBegin ? muMechanical : muLambda;
    //         augmented(rows + col, col) = std::sqrt(mu);
    //     }

    //     const Eigen::VectorXd correction = augmented.colPivHouseholderQr().solve(augmentedRhs);

    //     for (Eigen::Index i = 0; i < cols; ++i)
    //         solution->set(i, static_cast<SReal>(correction[i]));

    //     system->dispatchSystemSolution(m_correction);
    // }

    /**
     * Multiplier-only LM fallback. This does not alter the residual or the stored
     * Jacobian. It solves an augmented least-squares system only when requested
     * by the outer Newton globalization.
     */
    bool solveLevenbergMarquardt(SReal muLambda) override
    {
        SCOPED_TIMER("NCPLMSolve");

        if (!(muLambda > 0_sreal) || !std::isfinite(static_cast<double>(muLambda)))
            return false;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        if (!system)
            return false;

        system->setSystemSolution(m_correction);
        system->setRHS(m_residual);

        auto* matrix = system->getSystemBaseMatrix();
        auto* rhs = system->getSystemRHSBaseVector();
        auto* solution = system->getSystemSolutionBaseVector();

        if (!matrix || !rhs || !solution)
            return false;

        const Eigen::Index rows = matrix->rowSize();
        const Eigen::Index cols = matrix->colSize();
        const Eigen::Index lambdaDofs = static_cast<Eigen::Index>(lambdaDofCount());
        const Eigen::Index lambdaBegin = cols - lambdaDofs;

        if (rows <= 0 || cols <= 0 || rows != cols || lambdaDofs <= 0 || lambdaBegin < 0)
            return false;

        Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(rows + lambdaDofs, cols);
        Eigen::VectorXd augmentedRhs = Eigen::VectorXd::Zero(rows + lambdaDofs);

        for (Eigen::Index row = 0; row < rows; ++row)
        {
            augmentedRhs[row] = static_cast<double>(rhs->element(row));

            for (Eigen::Index col = 0; col < cols; ++col)
                augmented(row, col) = static_cast<double>(matrix->element(row, col));
        }

        const double sqrtMu = std::sqrt(static_cast<double>(muLambda));
        for (Eigen::Index local = 0; local < lambdaDofs; ++local)
            augmented(rows + local, lambdaBegin + local) = sqrtMu;

        const Eigen::VectorXd correction = augmented.colPivHouseholderQr().solve(augmentedRhs);
        if (!correction.allFinite())
            return false;

        for (Eigen::Index i = 0; i < cols; ++i)
            solution->set(i, static_cast<SReal>(correction[i]));

        system->dispatchSystemSolution(m_correction);
        return correctionIsFinite();
    }

    /** Compatibility path; the merit solver reconstructs trials from m_newtonState. */
    void updateGuessFromLinearSolution(SReal alpha) override
    {
        m_position.peq(m_correction, alpha);
        projectAndPropagate();
    }

    SReal squaredNormDx() override
    {
        return m_correction.dot(m_correction);
    }

    SReal squaredLastEvaluation() override
    {
        return m_position.dot(m_position);
    }

    /** Store/restore the nonlinear solve-entry state. */
    void storeSolveState() override
    {
        m_solveState.eq(m_position.id());
    }

    void restoreSolveState() override
    {
        restoreState(m_solveState);
    }

    /** Store/restore the current accepted Newton base. */
    void storeNewtonState() override
    {
        m_newtonState.eq(m_position.id());
        m_cachedLambdaPredictor.clear();
    }

    void restoreNewtonState() override
    {
        restoreState(m_newtonState);
    }

    /** Construct z(alpha)=z_k+alpha*dz from the same accepted base every time. */
    void setTrialFromNewtonState(SReal alpha) override
    {
        m_position.eq(m_newtonState.id());
        m_position.peq(m_correction, alpha);
        projectAndPropagate();
    }

    NCPDebugResidualSummary currentNCPDebugSummary() const override
    {
        return m_summary;
    }

    NCPDebugDirectionSummary currentDirectionSummary() const override
    {
        NCPDebugDirectionSummary summary;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* solution = system ? system->getSystemSolutionBaseVector() : nullptr;

        if (!solution)
            return summary;

        const sofa::SignedIndex totalDofs = solution->size();
        const sofa::SignedIndex lambdaDofs = static_cast<sofa::SignedIndex>(lambdaDofCount());
        const sofa::SignedIndex mechanicalDofs = totalDofs - lambdaDofs;

        if (totalDofs <= 0 || lambdaDofs < 0 || mechanicalDofs < 0)
            return summary;

        // Current NCP scene assumption:
        // one leading Rigid3 mechanical block: 6 scalar DOFs per node,
        // followed by one Vec1 multiplier DOF per contact row.
        if (mechanicalDofs % 6 != 0)
            return summary;

        SReal totalSquaredNorm = 0_sreal;
        SReal translationSquaredNorm = 0_sreal;
        SReal rotationSquaredNorm = 0_sreal;
        SReal lambdaSquaredNorm = 0_sreal;

        const sofa::SignedIndex rigidNodes = mechanicalDofs / 6;

        for (sofa::SignedIndex node = 0; node < rigidNodes; ++node)
        {
            const sofa::SignedIndex offset = 6 * node;

            SReal nodeTranslationSquaredNorm = 0_sreal;
            SReal nodeRotationSquaredNorm = 0_sreal;

            for (sofa::SignedIndex component = 0; component < 3; ++component)
            {
                const SReal value = solution->element(offset + component);

                if (!std::isfinite(static_cast<double>(value)))
                    return NCPDebugDirectionSummary{};

                nodeTranslationSquaredNorm += value * value;
            }

            for (sofa::SignedIndex component = 3; component < 6; ++component)
            {
                const SReal value = solution->element(offset + component);

                if (!std::isfinite(static_cast<double>(value)))
                    return NCPDebugDirectionSummary{};

                nodeRotationSquaredNorm += value * value;
            }

            translationSquaredNorm += nodeTranslationSquaredNorm;
            rotationSquaredNorm += nodeRotationSquaredNorm;

            summary.maximumNodeTranslation =
                std::max(summary.maximumNodeTranslation, std::sqrt(nodeTranslationSquaredNorm));

            summary.maximumNodeRotation =
                std::max(summary.maximumNodeRotation, std::sqrt(nodeRotationSquaredNorm));
        }

        for (sofa::SignedIndex i = mechanicalDofs; i < totalDofs; ++i)
        {
            const SReal value = solution->element(i);

            if (!std::isfinite(static_cast<double>(value)))
                return NCPDebugDirectionSummary{};

            lambdaSquaredNorm += value * value;
            summary.maximumAbsLambda = std::max(summary.maximumAbsLambda, std::abs(value));
        }

        totalSquaredNorm =
            translationSquaredNorm
            + rotationSquaredNorm
            + lambdaSquaredNorm;

        summary.totalNorm = std::sqrt(totalSquaredNorm);
        summary.translationNorm = std::sqrt(translationSquaredNorm);
        summary.rotationNorm = std::sqrt(rotationSquaredNorm);
        summary.lambdaNorm = std::sqrt(lambdaSquaredNorm);

        summary.rigidNodes = static_cast<sofa::Size>(rigidNodes);
        summary.lambdaDofs = static_cast<sofa::Size>(lambdaDofs);
        summary.valid = true;

        return summary;
    }

    bool correctionIsFinite() const override
    {
        const SReal norm2 = m_correction.dot(m_correction);
        return std::isfinite(static_cast<double>(norm2)) && norm2 >= 0_sreal;
    }

    void beginFiniteDifferenceCheck() override
    {
        m_fdReady = false;
        m_fdDetailedLogged = false;

        if (!m_finiteDifferenceCheck)
            return;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* matrix = system ? system->getSystemBaseMatrix() : nullptr;
        auto* rhs = system ? system->getSystemRHSBaseVector() : nullptr;
        auto* solution = system ? system->getSystemSolutionBaseVector() : nullptr;

        if (!matrix || !rhs || !solution)
            return;

        const sofa::SignedIndex rows = matrix->rowSize();
        const sofa::SignedIndex cols = matrix->colSize();
        if (rows <= 0 || rows != cols || rhs->size() != rows || solution->size() != cols)
            return;

        const sofa::Size lambdaDofs = lambdaDofCount();
        if (lambdaDofs > static_cast<sofa::Size>(rows))
            return;

        m_fdMechanicalDofs = static_cast<sofa::Size>(rows) - lambdaDofs;
        m_fdBaseResidual.resize(static_cast<std::size_t>(rows));
        m_fdJd.resize(static_cast<std::size_t>(rows));

        sofa::linearalgebra::FullVector<double> direction(cols);
        sofa::linearalgebra::FullVector<double> Ad(rows);
        for (sofa::SignedIndex col = 0; col < cols; ++col)
            direction[col] = static_cast<double>(solution->element(col));

        matrix->opMulV(&Ad, &direction);

        SReal linearDefect2 = 0_sreal;
        SReal rhs2 = 0_sreal;

        for (sofa::SignedIndex row = 0; row < rows; ++row)
        {
            const SReal base = rhs->element(row);
            const SReal matrixDirection = static_cast<SReal>(Ad[row]);

            m_fdBaseResidual[static_cast<std::size_t>(row)] = base;
            m_fdJd[static_cast<std::size_t>(row)] = -matrixDirection; // A=-J.

            const SReal defect = matrixDirection - base; // linear system is A*d=R.
            linearDefect2 += defect * defect;
            rhs2 += base * base;
        }

        const SReal denom = std::max(std::sqrt(rhs2), std::numeric_limits<SReal>::epsilon());
        const SReal linearRelErr = std::sqrt(linearDefect2) / denom;

        // A*d=R should be solved to high accuracy. Stay silent for healthy solves.
        if (linearRelErr > 1e-8_sreal)
        {
            msg_warning(&m_owner) << "[NCP FD LINEAR]"
                                  << " rows=" << rows
                                  << " M=" << m_fdMechanicalDofs
                                  << " FB=" << lambdaDofs
                                  << " relErr=" << linearRelErr;
        }

        m_contact->storeFiniteDifferenceBase();
        m_fdReady = true;
    }

    void evaluateFiniteDifferenceTrial(SReal alpha, unsigned int iteration, unsigned int trial) override
    {
        if (!m_finiteDifferenceCheck || !m_fdReady || !(alpha > 0_sreal))
            return;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        if (!system)
            return;

        // Flatten the already evaluated true trial residual in the same global ordering.
        system->setRHS(m_residual);
        auto* rhs = system->getSystemRHSBaseVector();
        if (!rhs || static_cast<std::size_t>(rhs->size()) != m_fdBaseResidual.size())
            return;

        SReal error2 = 0_sreal;
        SReal fd2 = 0_sreal;
        SReal predicted2 = 0_sreal;
        SReal mechanicalError2 = 0_sreal;
        SReal mechanicalFD2 = 0_sreal;
        SReal mechanicalPredicted2 = 0_sreal;
        SReal complementarityError2 = 0_sreal;
        SReal complementarityFD2 = 0_sreal;
        SReal complementarityPredicted2 = 0_sreal;

        struct RowErrorCandidate
        {
            sofa::SignedIndex row = -1;
            SReal fd = 0_sreal;
            SReal predicted = 0_sreal;
            SReal error = 0_sreal;
            SReal relativeError = 0_sreal;
            SReal score = 0_sreal;
            bool signFlip = false;
        };

        std::vector<RowErrorCandidate> mechanicalCandidates;
        std::vector<RowErrorCandidate> fbCandidates;

        // Row-level diagnostics are intentionally emitted only once per Newton
        // direction and only once the line-search step is genuinely local.
        static constexpr SReal localAlphaThreshold = 1e-4_sreal;
        static constexpr SReal localTranslationThreshold = 1e-3_sreal; // mm
        static constexpr SReal relativeTolerance = 5e-2_sreal;
        static constexpr SReal mechanicalAbsoluteTolerance = 1e-6_sreal;
        static constexpr SReal fbAbsoluteTolerance = 1e-6_sreal;
        static constexpr std::size_t maxRowsPerBlock = 2;

        const NCPDebugDirectionSummary directionSummary = currentDirectionSummary();
        const SReal stepMaxDx = directionSummary.valid
            ? alpha * directionSummary.maximumNodeTranslation
            : std::numeric_limits<SReal>::infinity();
        const SReal stepMaxDlambda = directionSummary.valid
            ? alpha * directionSummary.maximumAbsLambda
            : std::numeric_limits<SReal>::infinity();

        const bool localDiagnostic = !m_fdDetailedLogged
            && (alpha <= localAlphaThreshold || stepMaxDx <= localTranslationThreshold);

        for (sofa::SignedIndex row = 0; row < rhs->size(); ++row)
        {
            const SReal base = m_fdBaseResidual[static_cast<std::size_t>(row)];
            const SReal current = rhs->element(row);
            const SReal fd = (current - base) / alpha;
            const SReal predicted = m_fdJd[static_cast<std::size_t>(row)];
            const SReal error = fd - predicted;
            const bool mechanical = static_cast<sofa::Size>(row) < m_fdMechanicalDofs;

            error2 += error * error;
            fd2 += fd * fd;
            predicted2 += predicted * predicted;

            if (mechanical)
            {
                mechanicalError2 += error * error;
                mechanicalFD2 += fd * fd;
                mechanicalPredicted2 += predicted * predicted;
            }
            else
            {
                complementarityError2 += error * error;
                complementarityFD2 += fd * fd;
                complementarityPredicted2 += predicted * predicted;
            }

            if (!localDiagnostic)
                continue;

            const SReal absTol = mechanical ? mechanicalAbsoluteTolerance : fbAbsoluteTolerance;
            const SReal scale = std::max(std::abs(fd), std::abs(predicted));
            const SReal allowedError = std::max(absTol, relativeTolerance * scale);
            const SReal absoluteError = std::abs(error);
            const SReal score = absoluteError / std::max(allowedError, std::numeric_limits<SReal>::epsilon());
            const SReal relativeError = absoluteError / std::max(scale, absTol);
            const bool signFlip = std::abs(fd) > absTol && std::abs(predicted) > absTol && fd * predicted < 0_sreal;

            if (score < 1_sreal && !signFlip)
                continue;

            RowErrorCandidate candidate;
            candidate.row = row;
            candidate.fd = fd;
            candidate.predicted = predicted;
            candidate.error = error;
            candidate.relativeError = relativeError;
            candidate.score = score;
            candidate.signFlip = signFlip;

            if (mechanical)
                mechanicalCandidates.push_back(candidate);
            else
                fbCandidates.push_back(candidate);
        }

        if (!localDiagnostic)
            return;

        const SReal eps = std::numeric_limits<SReal>::epsilon();
        const SReal relScale = std::max({std::sqrt(fd2), std::sqrt(predicted2), eps});
        const SReal mechScale = std::max({std::sqrt(mechanicalFD2), std::sqrt(mechanicalPredicted2), eps});
        const SReal fbScale = std::max({std::sqrt(complementarityFD2), std::sqrt(complementarityPredicted2), eps});
        const SReal relErr = std::sqrt(error2) / relScale;
        const SReal mechRelErr = std::sqrt(mechanicalError2) / mechScale;
        const SReal fbRelErr = std::sqrt(complementarityError2) / fbScale;

        auto sortCandidates = [](auto& candidates)
        {
            std::sort(candidates.begin(), candidates.end(), [](const RowErrorCandidate& a, const RowErrorCandidate& b)
            {
                if (a.signFlip != b.signFlip)
                    return a.signFlip > b.signFlip;
                return a.score > b.score;
            });
        };

        sortCandidates(mechanicalCandidates);
        sortCandidates(fbCandidates);

        const SReal worstMechanical = mechanicalCandidates.empty() ? 0_sreal : mechanicalCandidates.front().score;
        const SReal worstFB = fbCandidates.empty() ? 0_sreal : fbCandidates.front().score;
        const bool suspicious = mechRelErr > 0.1_sreal || fbRelErr > 0.1_sreal
                             || worstMechanical >= 1_sreal || worstFB >= 1_sreal;

        if (suspicious)
        {
            msg_warning(&m_owner) << "[NCP FD SUSPECT]"
                                  << " it=" << iteration
                                  << " trial=" << trial
                                  << " alpha=" << alpha
                                  << " stepDx=" << stepMaxDx
                                  << " stepDlambda=" << stepMaxDlambda
                                  << " global=" << relErr
                                  << " M=" << mechRelErr
                                  << " FB=" << fbRelErr
                                  << " Mbad=" << mechanicalCandidates.size()
                                  << " FBbad=" << fbCandidates.size()
                                  << " worstM=" << worstMechanical
                                  << " worstFB=" << worstFB;

            static constexpr const char* rigidLabels[6] = {"Fx", "Fy", "Fz", "Mx", "My", "Mz"};

            const std::size_t mechanicalCount = std::min(maxRowsPerBlock, mechanicalCandidates.size());
            for (std::size_t rank = 0; rank < mechanicalCount; ++rank)
            {
                const auto& c = mechanicalCandidates[rank];
                const sofa::SignedIndex node = c.row / 6;
                const sofa::SignedIndex component = c.row % 6;
                const SReal decades = std::log10(std::max(c.score, 1_sreal));

                msg_warning(&m_owner) << "[NCP FD TOP M]"
                                      << " rank=" << rank + 1
                                      << " row=" << c.row
                                      << " node=" << node
                                      << " dof=" << rigidLabels[component]
                                      << " score=" << c.score
                                      << " decades=" << decades
                                      << " rel=" << c.relativeError
                                      << " fd=" << c.fd
                                      << " Jd=" << c.predicted
                                      << " err=" << c.error
                                      << " signFlip=" << c.signFlip;
            }

            const std::size_t fbCount = std::min(maxRowsPerBlock, fbCandidates.size());
            for (std::size_t rank = 0; rank < fbCount; ++rank)
            {
                const auto& c = fbCandidates[rank];
                const sofa::SignedIndex contactIndex = c.row - static_cast<sofa::SignedIndex>(m_fdMechanicalDofs);
                const SReal decades = std::log10(std::max(c.score, 1_sreal));

                msg_warning(&m_owner) << "[NCP FD TOP FB]"
                                      << " rank=" << rank + 1
                                      << " row=" << c.row
                                      << " contact=" << contactIndex
                                      << " score=" << c.score
                                      << " decades=" << decades
                                      << " rel=" << c.relativeError
                                      << " fd=" << c.fd
                                      << " Jd=" << c.predicted
                                      << " err=" << c.error
                                      << " signFlip=" << c.signFlip;
            }

            // Contact-level diagnostics are expensive/noisy. Evaluate them only
            // for this one local suspicious trial; the contact logger itself
            // reports only the worst contact and aggregate SDF-quality metrics.
            m_contact->logFiniteDifferenceTrial(static_cast<ContactForceField::Real>(alpha));
        }

        m_fdDetailedLogged = true;
    }

    /**
     * Select the line-search direction exactly in the spirit of Algorithm 4.1.
     *
     * The static system assembled by this class is
     *
     *     A = -J,
     *
     * because setSystemMBKMatrix() uses stiffnessFactor=-1 while the nonlinear
     * residual is R(z). For
     *
     *     Psi(z) = 0.5 R(z)^T W^2 R(z),
     *
     * we therefore have
     *
     *     -grad(Psi) = -J^T W^2 R = A^T W^2 R.
     *
     * SOFA already stores both A and the flattened RHS R in the linear system,
     * so the fallback requires only one transpose matrix-vector product.
     *
     * IMPORTANT:
     * This is the true merit gradient only when the assembled J is the true
     * derivative of evaluateCurrentGuess(). This is exact for the affine plane
     * contact with compliance frozen during the nonlinear solve. For curved
     * geometries, add the missing lambda*Hess(g) and any dr/dx term first.
     */
    bool selectMeritSearchDirection(SReal rho,SReal exponent,SReal& meritSlope,bool& usedGradientFallback) override
    {
        SOFA_UNUSED(rho);
        SOFA_UNUSED(exponent);

        meritSlope = std::numeric_limits<SReal>::quiet_NaN();
        usedGradientFallback = false;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* matrix = system ? system->getSystemBaseMatrix() : nullptr;
        auto* rhs = system ? system->getSystemRHSBaseVector() : nullptr;
        auto* solution = system ? system->getSystemSolutionBaseVector() : nullptr;

        if (!system || !matrix || !rhs || !solution)
            return false;

        const auto rows = matrix->rowSize();
        const auto cols = matrix->colSize();

        if (rows <= 0
            || rows != cols
            || rhs->size() != rows
            || solution->size() != cols)
        {
            return false;
        }

        // One Vec1 multiplier DOF exists per contact row. In the present plugin
        // architecture the multiplier block is the trailing block of the global
        // system. The three status counts always sum to the number of rows.
        const auto diagnostics = m_contact->currentContactDiagnostics();
        const sofa::Size lambdaDofs = diagnostics.activeCount + diagnostics.pinnedCount + diagnostics.invalidCount;

        if (lambdaDofs > static_cast<sofa::Size>(rows))
            return false;

        const sofa::Size mechanicalDofs = static_cast<sofa::Size>(rows) - lambdaDofs;

        const SReal invMechanicalReference2 = 1_sreal / (m_mechanicalReference * m_mechanicalReference);
        const SReal invComplementarityReference2 = 1_sreal / (m_complementarityReference * m_complementarityReference);

        sofa::linearalgebra::FullVector<double> weightedResidual(rows);
        sofa::linearalgebra::FullVector<double> negativeGradient(cols);

        for (sofa::SignedIndex i = 0; i < rows; ++i)
        {
            const SReal weight = static_cast<sofa::Size>(i) < mechanicalDofs
                                    ? invMechanicalReference2
                                    : invComplementarityReference2;

            const SReal value = weight * rhs->element(i);
            if (!std::isfinite(static_cast<double>(value)))
                return false;

            weightedResidual[i] = static_cast<double>(value);
        }

        // A = -J, hence this is -grad(Psi).
        matrix->opMulTV(&negativeGradient, &weightedResidual);

        SReal gradientNorm2 = 0_sreal;
        SReal newtonNorm2 = 0_sreal;
        SReal negativeGradientDotNewton = 0_sreal;

        for (sofa::SignedIndex i = 0; i < cols; ++i)
        {
            const SReal minusGradient = static_cast<SReal>(negativeGradient[i]);
            const SReal newtonDirection = solution->element(i);

            if (!std::isfinite(static_cast<double>(minusGradient))
                || !std::isfinite(static_cast<double>(newtonDirection)))
            {
                return false;
            }

            gradientNorm2 += minusGradient * minusGradient;
            newtonNorm2 += newtonDirection * newtonDirection;
            negativeGradientDotNewton += minusGradient * newtonDirection;
        }

        if (!std::isfinite(static_cast<double>(gradientNorm2))
            || !std::isfinite(static_cast<double>(newtonNorm2)))
        {
            return false;
        }

        // Since negativeGradient=-grad(Psi):
        //
        //     grad(Psi)^T d_N = -negativeGradient^T d_N.
        const SReal newtonMeritSlope = -negativeGradientDotNewton;
        const SReal newtonNorm = std::sqrt(std::max(newtonNorm2, 0_sreal));
        // const SReal requiredSlope = -rho * std::pow(newtonNorm, exponent);
        const SReal requiredSlope = 0.0;

        if (m_debug)
        {
            msg_info(&m_owner) << "[NCP DESCENT]"
                               << " slope=" << newtonMeritSlope
                               << " |d|=" << newtonNorm
                               << " Criterion=" << requiredSlope
                               << " |gradPsi|=" << std::sqrt(gradientNorm2)
                               << " descent=" << (newtonMeritSlope < 0_sreal);
        }

        if (std::isfinite(static_cast<double>(newtonMeritSlope)) && newtonMeritSlope <= requiredSlope)
        {
            meritSlope = newtonMeritSlope;
            usedGradientFallback = false;

            // The current system solution already contains d_N and was already
            // dispatched to m_correction by solveLinearEquation().
            return true;
        }

        // Newton direction does not have enough descent. Use
        //
        //     d_G = -grad(Psi).
        if (!(gradientNorm2 > std::numeric_limits<SReal>::epsilon()))
            return false;

        for (sofa::SignedIndex i = 0; i < cols; ++i)
            solution->set(i, static_cast<SReal>(negativeGradient[i]));

        system->dispatchSystemSolution(m_correction);

        if (m_debug)
        {
            msg_info(&m_owner) << "[NCP FALLBACK]"
                               << " |gradPsi|=" << std::sqrt(gradientNorm2)
                               << " slope=" << -gradientNorm2
                               << " |d|=" << std::sqrt(m_correction.dot(m_correction));
        }

        meritSlope = -gradientNorm2;
        usedGradientFallback = correctionIsFinite() && std::isfinite(static_cast<double>(meritSlope)) && meritSlope < 0_sreal;
        msg_warning(&m_owner) << "[NCP Gradient Fallback]"
                            << " Newton=" << newtonMeritSlope
                            << " Criterion=" << requiredSlope
                             << " gradPsi=" << -gradientNorm2;
        return usedGradientFallback;
    }

    SReal maxAbsLambdaCorrection() const override
    {
        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* solution = system ? system->getSystemSolutionBaseVector() : nullptr;
        if (!solution)
            return std::numeric_limits<SReal>::infinity();

        const sofa::SignedIndex lambdaDofs = static_cast<sofa::SignedIndex>(lambdaDofCount());
        const sofa::SignedIndex size = solution->size();
        if (lambdaDofs <= 0 || lambdaDofs > size)
            return std::numeric_limits<SReal>::infinity();

        SReal maxValue = 0_sreal;
        for (sofa::SignedIndex i = size - lambdaDofs; i < size; ++i)
            maxValue = std::max(maxValue, std::abs(solution->element(i)));

        return maxValue;
    }

    bool cacheLambdaPredictor() override
    {
        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* solution = system ? system->getSystemSolutionBaseVector() : nullptr;
        if (!solution)
            return false;

        const sofa::SignedIndex lambdaDofs = static_cast<sofa::SignedIndex>(lambdaDofCount());
        const sofa::SignedIndex size = solution->size();
        const sofa::SignedIndex lambdaBegin = size - lambdaDofs;
        if (lambdaDofs <= 0 || lambdaBegin < 0)
            return false;

        m_cachedLambdaPredictor.resize(static_cast<std::size_t>(lambdaDofs));

        for (sofa::SignedIndex i = 0; i < lambdaDofs; ++i)
        {
            const SReal value = solution->element(lambdaBegin + i);
            if (!std::isfinite(static_cast<double>(value)))
                return false;

            m_cachedLambdaPredictor[static_cast<std::size_t>(i)] = value;
        }

        return true;
    }

    void storeContinuationState() override
    {
        m_continuationState.eq(m_position.id());
    }

    void restoreContinuationState() override
    {
        restoreState(m_continuationState);
        evaluateCurrentGuess();
    }

    bool applyLambdaPredictorFraction(SReal fraction, SReal& appliedInfinityNorm) override
    {
        appliedInfinityNorm = 0_sreal;

        if (!(fraction > 0_sreal) || fraction > 1_sreal || m_cachedLambdaPredictor.empty())
            return false;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* solution = system ? system->getSystemSolutionBaseVector() : nullptr;
        if (!solution)
            return false;

        const sofa::SignedIndex lambdaDofs = static_cast<sofa::SignedIndex>(m_cachedLambdaPredictor.size());
        const sofa::SignedIndex size = solution->size();
        const sofa::SignedIndex lambdaBegin = size - lambdaDofs;
        if (lambdaDofs <= 0 || lambdaBegin < 0)
            return false;

        for (sofa::SignedIndex i = 0; i < lambdaBegin; ++i)
            solution->set(i, 0_sreal);

        for (sofa::SignedIndex i = 0; i < lambdaDofs; ++i)
        {
            const SReal increment = fraction * m_cachedLambdaPredictor[static_cast<std::size_t>(i)];
            solution->set(lambdaBegin + i, increment);
            appliedInfinityNorm = std::max(appliedInfinityNorm, std::abs(increment));
        }

        system->dispatchSystemSolution(m_correction);

        // Important: increment the CURRENT accepted continuation state. Do not
        // reconstruct from m_newtonState; m_newtonState is the fixed outer base.
        m_position.peq(m_correction, 1_sreal);
        projectAndPropagate();
        evaluateCurrentGuess();

        return m_summary.valid;
    }

    void storeMechanicalState() override
    {
        m_mechanicalState.eq(m_position.id());
    }

    void restoreMechanicalState() override
    {
        restoreState(m_mechanicalState);
        evaluateCurrentGuess();
    }

    bool solveMechanicalLinearEquation() override
    {
        SCOPED_TIMER("NCPMechanicalLinearSolve");

        assembleLinearizedSystem(false);

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        if (!system)
            return false;

        system->setSystemSolution(m_correction);
        system->setRHS(m_residual);

        auto* matrix = system->getSystemBaseMatrix();
        auto* rhs = system->getSystemRHSBaseVector();
        auto* solution = system->getSystemSolutionBaseVector();
        if (!matrix || !rhs || !solution)
            return false;

        const sofa::SignedIndex lambdaDofs = static_cast<sofa::SignedIndex>(lambdaDofCount());
        const sofa::SignedIndex size = rhs->size();
        const sofa::SignedIndex lambdaBegin = size - lambdaDofs;

        if (lambdaDofs <= 0 || lambdaBegin < 0 || matrix->rowSize() != size || matrix->colSize() != size)
            return false;

        /**
         * Fixed-lambda mechanical Newton:
         *
         * [ A_xx  A_xλ ] [dx]   [R_x]
         * [   0     I  ] [dλ] = [ 0 ].
         *
         * Only lambda ROWS are replaced. A_xλ is deliberately kept: the first
         * residual equation still contains the current H^T lambda force, while
         * dλ=0 makes A_xλ*dλ vanish in this inner correction.
         */
        matrix->clearRows(lambdaBegin, size);
        for (sofa::SignedIndex i = lambdaBegin; i < size; ++i)
        {
            matrix->set(i, i, 1.0);
            rhs->set(i, 0_sreal);
        }

        matrix->compress();
        m_linearSolver->solveSystem();

        // Enforce the mathematical constraint exactly before dispatch, even if
        // the linear solver leaves tiny round-off values in the identity rows.
        for (sofa::SignedIndex i = lambdaBegin; i < size; ++i)
            solution->set(i, 0_sreal);

        system->dispatchSystemSolution(m_correction);
        return correctionIsFinite();
    }

    void setMechanicalTrialFromState(SReal alpha) override
    {
        m_position.eq(m_mechanicalState.id());
        m_position.peq(m_correction, alpha);
        projectAndPropagate();
    }

    void evaluateMechanicalTrial() override
    {
        evaluateCurrentGuess();
    }

    SReal scaledMechanicalSquaredNorm() const override
    {
        return m_summary.scaledMechanicalSquaredNorm;
    }

    bool continuationStateValid() const override
    {
        return m_summary.valid && std::isfinite(static_cast<double>(m_summary.scaledMechanicalSquaredNorm));
    }

    SReal mechanicalResidualNorm() const
    {
        return std::sqrt(std::max(m_mechanicalResidualSquaredNorm, SReal(0)));
    }

    SReal complementarityResidualNorm() const
    {
        return std::sqrt(std::max(m_complementarityResidualSquaredNorm, SReal(0)));
    }

private:
    NCPStaticSolver& m_owner;
    MechanicalOperations& m_mechanicalOperations;
    core::behavior::MultiVecCoord& m_position;
    core::behavior::MultiVecDeriv& m_residual;
    core::behavior::MultiVecDeriv& m_correction;
    core::behavior::LinearSolver* m_linearSolver = nullptr;
    ContactForceField* m_contact = nullptr;
    core::behavior::MultiVecCoord m_solveState;
    core::behavior::MultiVecCoord m_newtonState;
    core::behavior::MultiVecCoord m_continuationState;
    core::behavior::MultiVecCoord m_mechanicalState;
    std::vector<SReal> m_cachedLambdaPredictor;
    const SReal m_mechanicalReference;
    const SReal m_complementarityReference;
    const bool m_debug;
    const bool m_finiteDifferenceCheck;

    std::vector<SReal> m_fdBaseResidual;
    std::vector<SReal> m_fdJd;
    sofa::Size m_fdMechanicalDofs = 0;
    bool m_fdReady = false;
    bool m_fdDetailedLogged = false;

    NCPDebugResidualSummary m_summary;
    SReal m_scaledResidualSquaredNorm = std::numeric_limits<SReal>::infinity();
    SReal m_mechanicalResidualSquaredNorm = std::numeric_limits<SReal>::infinity();
    SReal m_complementarityResidualSquaredNorm = std::numeric_limits<SReal>::infinity();

    sofa::Size lambdaDofCount() const
    {
        const auto diagnostics = m_contact->currentContactDiagnostics();
        return diagnostics.activeCount + diagnostics.pinnedCount + diagnostics.invalidCount;
    }

    bool applyAdaptiveContactRegularization()
    {
        m_owner.d_lastContactSchurSigmaMinBefore.setValue(0_sreal);
        m_owner.d_lastContactSchurSigmaMinAfter.setValue(0_sreal);
        m_owner.d_lastContactSchurConditionBefore.setValue(0_sreal);
        m_owner.d_lastAdaptiveContactRegularization.setValue(0_sreal);

        if (!m_owner.d_adaptiveContactRegularization.getValue())
            return true;

        auto* system = m_linearSolver ? m_linearSolver->getLinearSystem() : nullptr;
        auto* matrix = system ? system->getSystemBaseMatrix() : nullptr;
        if (!matrix)
            return false;

        const sofa::SignedIndex rows = matrix->rowSize();
        const sofa::SignedIndex cols = matrix->colSize();
        const sofa::SignedIndex lambdaDofs = static_cast<sofa::SignedIndex>(lambdaDofCount());
        const sofa::SignedIndex mechanicalDofs = rows - lambdaDofs;
        const sofa::SignedIndex lambdaBegin = mechanicalDofs;

        if (rows <= 0 || rows != cols || lambdaDofs <= 0 || mechanicalDofs <= 0)
            return true;

        // Prefer the exact current contact status when point-indexed debug data is
        // being published. If publishing is disabled, infer the coupled rows from
        // the already assembled mixed matrix. Pinned/invalid rows have H=0 and
        // therefore no x-lambda coupling.
        std::vector<sofa::SignedIndex> activeLocalIndices;
        activeLocalIndices.reserve(static_cast<std::size_t>(lambdaDofs));

        const auto& publishedStatus = m_contact->d_contactStatus.getValue();
        const bool havePublishedStatus = m_contact->d_publishContactData.getValue()
            && publishedStatus.size() == static_cast<std::size_t>(lambdaDofs);

        if (havePublishedStatus)
        {
            for (sofa::SignedIndex local = 0; local < lambdaDofs; ++local)
            {
                if (publishedStatus[static_cast<std::size_t>(local)]
                    == static_cast<unsigned int>(ContactRowStatus::Active))
                {
                    activeLocalIndices.push_back(local);
                }
            }
        }
        else
        {
            static constexpr SReal couplingTolerance2 = 1e-24_sreal;
            static constexpr SReal pinnedDiagonalTolerance = 1e-6_sreal;

            for (sofa::SignedIndex local = 0; local < lambdaDofs; ++local)
            {
                const sofa::SignedIndex global = lambdaBegin + local;
                SReal coupling2 = 0_sreal;

                for (sofa::SignedIndex i = 0; i < mechanicalDofs; ++i)
                {
                    const SReal upper = matrix->element(i, global);
                    const SReal lower = matrix->element(global, i);
                    coupling2 += upper * upper + lower * lower;
                }

                // A=-J. A pinned row has A_ll=-1 and no coupling. The diagonal
                // fallback also catches an active row whose projected coupling is
                // accidentally zero but whose FB diagonal differs from the pinned row.
                const SReal jacobianDiagonal = -matrix->element(global, global);
                const bool coupled = coupling2 > couplingTolerance2;
                const bool nonPinnedDiagonal = std::abs(jacobianDiagonal - 1_sreal) > pinnedDiagonalTolerance;

                if (coupled || nonPinnedDiagonal)
                    activeLocalIndices.push_back(local);
            }
        }

        const Eigen::Index activeCount = static_cast<Eigen::Index>(activeLocalIndices.size());
        if (activeCount == 0)
            return true;

        const Eigen::Index nMechanical = static_cast<Eigen::Index>(mechanicalDofs);

        Eigen::MatrixXd Axx(nMechanical, nMechanical);
        Eigen::MatrixXd B(nMechanical, activeCount);
        Eigen::MatrixXd C(activeCount, nMechanical);
        Eigen::MatrixXd D(activeCount, activeCount);

        for (Eigen::Index row = 0; row < nMechanical; ++row)
        {
            for (Eigen::Index col = 0; col < nMechanical; ++col)
                Axx(row, col) = static_cast<double>(matrix->element(row, col));
        }

        for (Eigen::Index j = 0; j < activeCount; ++j)
        {
            const sofa::SignedIndex lambdaLocal = activeLocalIndices[static_cast<std::size_t>(j)];
            const sofa::SignedIndex lambdaGlobal = lambdaBegin + lambdaLocal;

            for (Eigen::Index i = 0; i < nMechanical; ++i)
            {
                B(i, j) = static_cast<double>(matrix->element(i, lambdaGlobal));
                C(j, i) = static_cast<double>(matrix->element(lambdaGlobal, i));
            }

            for (Eigen::Index k = 0; k < activeCount; ++k)
            {
                const sofa::SignedIndex otherLocal = activeLocalIndices[static_cast<std::size_t>(k)];
                const sofa::SignedIndex otherGlobal = lambdaBegin + otherLocal;
                D(j, k) = static_cast<double>(matrix->element(lambdaGlobal, otherGlobal));
            }
        }

        if (!Axx.allFinite() || !B.allFinite() || !C.allFinite() || !D.allFinite())
        {
            msg_warning(&m_owner) << "[NCP SCHUR REG] skipped: non-finite assembled block.";
            return false;
        }

        // A=-J. Form the active Schur complement of A and negate it to obtain
        // the reduced contact Jacobian:
        //
        //     S_A = D - C A_xx^{-1} B,
        //     S_J = -S_A.
        //
        // A scalar tangent regularization +gamma I in J_ll therefore appears as
        // -gamma I in the assembled A_ll block and shifts S_J by +gamma I.
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Axx);
        qr.setThreshold(1e-12);

        if (qr.rank() < nMechanical)
        {
            msg_warning(&m_owner) << "[NCP SCHUR REG] skipped: mechanical block rank="
                                  << qr.rank() << "/" << nMechanical
                                  << ". Contact-only damping cannot regularize a singular A_xx.";
            return false;
        }

        const Eigen::MatrixXd X = qr.solve(B);
        const double solveDenominator = std::max(B.norm(), std::numeric_limits<double>::epsilon());
        const double solveRelativeError = (Axx * X - B).norm() / solveDenominator;

        if (!X.allFinite() || !std::isfinite(solveRelativeError) || solveRelativeError > 1e-7)
        {
            msg_warning(&m_owner) << "[NCP SCHUR REG] skipped: A_xx solve relErr="
                                  << solveRelativeError << ".";
            return false;
        }

        const Eigen::MatrixXd schurJ = -(D - C * X);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdBefore(schurJ, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd singularValuesBefore = svdBefore.singularValues();
        if (singularValuesBefore.size() == 0 || !singularValuesBefore.allFinite())
            return false;

        const SReal sigmaMaxBefore = static_cast<SReal>(singularValuesBefore[0]);
        const SReal sigmaMinBefore = static_cast<SReal>(singularValuesBefore[singularValuesBefore.size() - 1]);
        const SReal sigmaDenominator = std::max(sigmaMinBefore, std::numeric_limits<SReal>::epsilon());
        const SReal conditionBefore = sigmaMaxBefore / sigmaDenominator;

        m_owner.d_lastContactSchurSigmaMinBefore.setValue(sigmaMinBefore);
        m_owner.d_lastContactSchurSigmaMinAfter.setValue(sigmaMinBefore);
        m_owner.d_lastContactSchurConditionBefore.setValue(conditionBefore);

        const SReal sigmaFloor = m_owner.d_contactSchurSingularValueFloor.getValue();
        const SReal targetCondition = m_owner.d_contactSchurTargetConditionNumber.getValue();
        const bool unsafeSigma = sigmaMinBefore < sigmaFloor;
        const bool unsafeCondition = targetCondition > 1_sreal && conditionBefore > targetCondition;

        if (!unsafeSigma && !unsafeCondition)
        {
            if (m_owner.d_logAdaptiveContactRegularization.getValue())
            {
                msg_info(&m_owner) << "[NCP SCHUR REG] active=" << activeCount
                                   << " sigmaMin=" << sigmaMinBefore
                                   << " sigmaMax=" << sigmaMaxBefore
                                   << " cond=" << conditionBefore
                                   << " addGamma=0";
            }
            return true;
        }

        // Closed-form initial shift from the symmetric part. If
        //
        //     H = 0.5 (S_J + S_J^T),
        //
        // then H(gamma)=H+gamma I, so its eigenvalues shift exactly by gamma.
        // Enforcing lambda_min(H+gamma I)>=sigmaFloor is a sufficient (not
        // necessary) condition for sigma_min(S_J+gamma I)>=sigmaFloor.
        const Eigen::MatrixXd symmetricPart = 0.5 * (schurJ + schurJ.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(symmetricPart, Eigen::EigenvaluesOnly);
        if (eigenSolver.info() != Eigen::Success || !eigenSolver.eigenvalues().allFinite())
            return false;

        const SReal lambdaMin = static_cast<SReal>(eigenSolver.eigenvalues()[0]);
        const SReal lambdaMax = static_cast<SReal>(eigenSolver.eigenvalues()[eigenSolver.eigenvalues().size() - 1]);

        SReal additionalGamma = std::max(0_sreal, sigmaFloor - lambdaMin);

        if (targetCondition > 1_sreal)
        {
            const SReal numerator = lambdaMax - targetCondition * lambdaMin;
            if (numerator > 0_sreal)
            {
                const SReal conditionShift = numerator / (targetCondition - 1_sreal);
                additionalGamma = std::max(additionalGamma, conditionShift);
            }
        }

        // The symmetric-part formula can be conservative for a nonsymmetric
        // Schur block. Verify the actual singular values after the proposed shift.
        // If needed, increase gamma geometrically until the requested SVD criteria
        // are met or the user cap is reached.
        const SReal maxAdditionalGamma = m_owner.d_contactSchurMaxAdditionalRegularization.getValue();
        additionalGamma = std::min(additionalGamma, maxAdditionalGamma);

        auto evaluateShift = [&](SReal gamma, SReal& sigmaMin, SReal& condition)
        {
            Eigen::MatrixXd shifted = schurJ;
            shifted.diagonal().array() += static_cast<double>(gamma);

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(shifted, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::VectorXd singularValues = svd.singularValues();

            if (singularValues.size() == 0 || !singularValues.allFinite())
                return false;

            const SReal sigmaMax = static_cast<SReal>(singularValues[0]);
            sigmaMin = static_cast<SReal>(singularValues[singularValues.size() - 1]);
            condition = sigmaMax / std::max(sigmaMin, std::numeric_limits<SReal>::epsilon());
            return true;
        };

        SReal sigmaMinAfter = sigmaMinBefore;
        SReal conditionAfter = conditionBefore;

        if (additionalGamma > 0_sreal)
            evaluateShift(additionalGamma, sigmaMinAfter, conditionAfter);

        auto shiftIsSafe = [&]()
        {
            const bool sigmaSafe = sigmaMinAfter >= sigmaFloor;
            const bool conditionSafe = targetCondition <= 1_sreal || conditionAfter <= targetCondition;
            return sigmaSafe && conditionSafe;
        };

        for (unsigned int attempt = 0; attempt < 8u && !shiftIsSafe() && additionalGamma < maxAdditionalGamma; ++attempt)
        {
            const SReal nextGamma = std::min(
                maxAdditionalGamma,
                std::max(2_sreal * additionalGamma, std::max(sigmaFloor, 1e-12_sreal)));

            if (!(nextGamma > additionalGamma))
                break;

            additionalGamma = nextGamma;
            if (!evaluateShift(additionalGamma, sigmaMinAfter, conditionAfter))
                return false;
        }

        if (!(additionalGamma > 0_sreal))
        {
            msg_warning(&m_owner) << "[NCP SCHUR REG] unsafe contact Schur block but additional regularization is capped at zero.";
            return false;
        }

        // The contact force field contributes +gamma to J_ll, while SOFA solves
        // A=-J. Therefore increasing the tangent-only regularization by deltaGamma
        // means subtracting deltaGamma from the assembled active lambda diagonal.
        for (const sofa::SignedIndex local : activeLocalIndices)
        {
            const sofa::SignedIndex global = lambdaBegin + local;
            matrix->set(global, global, matrix->element(global, global) - additionalGamma);
        }

        matrix->compress();

        m_owner.d_lastContactSchurSigmaMinAfter.setValue(sigmaMinAfter);
        m_owner.d_lastAdaptiveContactRegularization.setValue(additionalGamma);

        const bool capped = !shiftIsSafe();

        if (capped)
        {
            msg_warning(&m_owner) << "[NCP SCHUR REG] active=" << activeCount
                                  << " sigmaMin=" << sigmaMinBefore
                                  << " cond=" << conditionBefore
                                  << " symEigMin=" << lambdaMin
                                  << " symEigMax=" << lambdaMax
                                  << " baseGamma=" << m_contact->d_contactNewtonRegularization.getValue()
                                  << " addGamma=" << additionalGamma
                                  << " totalGamma=" << m_contact->d_contactNewtonRegularization.getValue() + additionalGamma
                                  << " sigmaMinAfter=" << sigmaMinAfter
                                  << " condAfter=" << conditionAfter
                                  << " capped=1";
        }
        else if (m_owner.d_logAdaptiveContactRegularization.getValue())
        {
            msg_info(&m_owner) << "[NCP SCHUR REG] active=" << activeCount
                               << " sigmaMin=" << sigmaMinBefore
                               << " cond=" << conditionBefore
                               << " symEigMin=" << lambdaMin
                               << " symEigMax=" << lambdaMax
                               << " baseGamma=" << m_contact->d_contactNewtonRegularization.getValue()
                               << " addGamma=" << additionalGamma
                               << " totalGamma=" << m_contact->d_contactNewtonRegularization.getValue() + additionalGamma
                               << " sigmaMinAfter=" << sigmaMinAfter
                               << " condAfter=" << conditionAfter
                               << " capped=0";
        }

        return !capped;
    }

    void assembleLinearizedSystem(bool applyAdaptiveRegularization = true)
    {
        SCOPED_TIMER("NCPJacobianAssembly");
        static constexpr core::MatricesFactors::M massFactor(0);
        static constexpr core::MatricesFactors::B dampingFactor(0);
        static constexpr core::MatricesFactors::K stiffnessFactor(-1);
        m_mechanicalOperations.setSystemMBKMatrix(massFactor,dampingFactor,stiffnessFactor,m_linearSolver);

        if (applyAdaptiveRegularization)
            applyAdaptiveContactRegularization();
    }

    /** Checkpoints are already projected; restoration only propagates mappings. */
    void restoreState(const core::behavior::MultiVecCoord& state)
    {
        m_position.eq(state.id());
        m_mechanicalOperations.propagateX(m_position);
    }

    void projectAndPropagate()
    {
        m_mechanicalOperations.solveConstraint(m_position,core::ConstraintOrder::POS);
        m_mechanicalOperations.propagateX(m_position);
    }
};

} // namespace

void NCPStaticSolver::solve(const core::ExecParams* params,SReal dt,core::MultiVecCoordId xResult,core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
        return;

    SOFA_UNUSED(dt);
    SOFA_UNUSED(vResult);

    sofa::simulation::common::VectorOperations vectorOperations(params, getContext());
    sofa::simulation::common::MechanicalOperations mechanicalOperations(params, getContext());
    mechanicalOperations->setImplicit(true);

    core::behavior::MultiVecCoord position(&vectorOperations, xResult);
    core::behavior::MultiVecDeriv residual(&vectorOperations, core::vec_id::write_access::force);
    core::behavior::MultiVecDeriv correction(&vectorOperations, core::vec_id::write_access::dx);
    correction.realloc(&vectorOperations, true, true);

    auto prepareCheckpoint = [&](core::MultiVecCoordId& id, const char* name)
    {
        core::behavior::MultiVecCoord state(&vectorOperations, id);
        state.realloc( &vectorOperations, true, true, core::VecIdProperties(name, getClassName()));
        id = state.id();
    };

    prepareCheckpoint(m_solveStateId, "ncpSolveState");
    prepareCheckpoint(m_newtonStateId, "ncpNewtonState");
    prepareCheckpoint(m_continuationStateId, "ncpContinuationState");
    prepareCheckpoint(m_mechanicalStateId, "ncpMechanicalState");

    auto* contact = l_contactForceField.get();
    contact->beginNonlinearSolve();

    NCPStaticResidualFunction residualFunction(
        *this,
        mechanicalOperations,
        position,
        residual,
        correction,
        l_linearSolver.get(),
        l_contactForceField.get(),
        m_solveStateId,
        m_newtonStateId,
        m_continuationStateId,
        m_mechanicalStateId,
        d_mechanicalResidualReference.getValue(),
        d_complementarityResidualReference.getValue(),
        d_debug.getValue(),
        d_finiteDifferenceCheck.getValue());

    residualFunction.projectCurrentState();

    auto* newton = dynamic_cast<NCPDebugNewtonRaphsonSolver*>(l_newtonSolver.get());

    if (auto* continuationNewton = dynamic_cast<NCPMechanicalContinuationNewtonRaphsonSolver*>(newton))
        continuationNewton->solveNCPContinuation(residualFunction);
    else
        newton->solveNCP(residualFunction);

    d_lastMechanicalResidualNorm.setValue(residualFunction.mechanicalResidualNorm());
    d_lastComplementarityResidualNorm.setValue(residualFunction.complementarityResidualNorm());

    bool complianceReady = false;
    if (newton->lastSolveConverged())
        complianceReady = contact->commitLaggedComplianceSnapshot();
    else
        contact->discardLaggedComplianceSnapshot();

    d_lastComplianceCaptureSucceeded.setValue(complianceReady);
}

} // namespace sofa::ncp
