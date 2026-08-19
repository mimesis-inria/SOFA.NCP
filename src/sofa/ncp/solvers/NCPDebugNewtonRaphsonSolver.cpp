#include <sofa/ncp/solvers/NCPDebugNewtonRaphsonSolver.h>

#include <sofa/component/odesolver/backward/NonLinearFunction.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/logging/Messaging.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>

namespace sofa::ncp
{

using sofa::component::odesolver::backward::NewtonStatus;
using BaseNonLinearFunction = sofa::component::odesolver::backward::newton_raphson::BaseNonLinearFunction;

void registerNCPDebugNewtonRaphsonSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData(
        "NCP Newton solver with Algorithm-4.1 merit descent safeguard and non-monotone Armijo backtracking.")
        .add<NCPDebugNewtonRaphsonSolver>());
}

NCPDebugNewtonRaphsonSolver::NCPDebugNewtonRaphsonSolver()
    : Inherit()
    , d_logIterationSummary(initData(&d_logIterationSummary, false,
        "logIterationSummary", "Print one base/linear/terminal line per Newton iteration."))
    , d_logLineSearchTrials(initData(&d_logLineSearchTrials, false,
        "logLineSearchTrials", "Print one line per merit-function backtracking trial."))
    , d_nonMonotoneWindow(initData(&d_nonMonotoneWindow, 1u,
        "nonMonotoneWindow", "Number of recent accepted merit values used by the non-monotone Armijo condition."))
    , d_enableLMFallback(initData(&d_enableLMFallback, false,
        "enableLMFallback", "Use one-shot LM rescue steps after Newton line-search failure or repeated tiny accepted Newton steps."))
    , d_lmInitialDamping(initData(&d_lmInitialDamping, 1e-4_sreal,
        "lmInitialDamping", "Initial multiplier-only LM damping."))
    , d_lmMaxRetries(initData(&d_lmMaxRetries, 3u,
        "lmMaxRetries", "Maximum LM damping attempts at the same Newton base."))
    , d_lmStagnationIterations(initData(&d_lmStagnationIterations, 2u,
        "lmStagnationIterations", "Consecutive tiny/ineffective accepted Newton steps before one LM rescue step."))
    , d_lastAcceptedNewtonUpdates(initData(&d_lastAcceptedNewtonUpdates, 0u,
        "lastAcceptedNewtonUpdates", "Number of Newton updates retained by the last solve."))
    , d_lastAcceptedAlpha(initData(&d_lastAcceptedAlpha, 0_sreal,
        "lastAcceptedAlpha", "Step length of the most recently retained Newton update."))
    , d_lastFailureReason(initData(&d_lastFailureReason, std::string(),
        "lastFailureReason", "Failure stage and reason; empty after convergence."))
{
    static std::string groupDiagnostics{"NCP Debug"};
    d_logIterationSummary.setGroup(groupDiagnostics);
    d_logLineSearchTrials.setGroup(groupDiagnostics);
    d_nonMonotoneWindow.setGroup(groupDiagnostics);
    d_enableLMFallback.setGroup(groupDiagnostics);
    d_lmInitialDamping.setGroup(groupDiagnostics);
    d_lmMaxRetries.setGroup(groupDiagnostics);
    d_lmStagnationIterations.setGroup(groupDiagnostics);
    d_lastAcceptedNewtonUpdates.setGroup(groupDiagnostics);
    d_lastAcceptedAlpha.setGroup(groupDiagnostics);
    d_lastFailureReason.setGroup(groupDiagnostics);

    d_lastAcceptedNewtonUpdates.setReadOnly(true);
    d_lastAcceptedAlpha.setReadOnly(true);
    d_lastFailureReason.setReadOnly(true);
}

void NCPDebugNewtonRaphsonSolver::init()
{
    Inherit::init();

    const SReal coefficient = d_lineSearchCoefficient.getValue();
    if (!(coefficient > 0_sreal && coefficient < 1_sreal))
    {
        msg_error() << "lineSearchCoefficient must be strictly between 0 and 1.";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
    }

    if (d_enableLMFallback.getValue()
        && (!(d_lmInitialDamping.getValue() > 0_sreal) || d_lmMaxRetries.getValue() == 0u))
    {
        msg_error() << "LM fallback requires lmInitialDamping > 0 and lmMaxRetries > 0.";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
    }
}

void NCPDebugNewtonRaphsonSolver::reset()
{
    Inherit::reset();
    d_lastAcceptedNewtonUpdates.setValue(0u);
    d_lastAcceptedAlpha.setValue(0_sreal);
    d_lastFailureReason.setValue(std::string());
}

namespace
{

constexpr SReal armijoCoefficient = 1e-4_sreal;   // sigma
constexpr SReal descentRho = 1e-8_sreal;           // rho
constexpr SReal descentExponent = 2_sreal;        // p
constexpr SReal lmStagnationAlphaThreshold = 1e-3_sreal;
constexpr SReal lmStagnationResidualRatio = 0.99_sreal;

bool finiteNonNegative(const SReal value)
{
    return std::isfinite(static_cast<double>(value)) && value >= 0_sreal;
}

SReal normFromSquared(const SReal value)
{
    return std::sqrt(std::max(value, 0_sreal));
}

SReal normRatio(const SReal trialNorm, const SReal baseNorm)
{
    return trialNorm / std::max(baseNorm,1e-10_sreal);
}

bool convergedAtAbsoluteThreshold(const SReal squaredResidual, const SReal absoluteThreshold)
{
    return absoluteThreshold > 0_sreal && squaredResidual <= absoluteThreshold * absoluteThreshold;
}

bool convergedAtRelativeInitialThreshold(const SReal squaredResidual, const SReal initialSquaredResidual, const SReal relativeInitialThreshold)
{
    return relativeInitialThreshold > 0_sreal && initialSquaredResidual > std::numeric_limits<SReal>::epsilon() && squaredResidual <= relativeInitialThreshold * relativeInitialThreshold * initialSquaredResidual;
}

} // namespace

void NCPDebugNewtonRaphsonSolver::solveNCP(BaseNonLinearFunction& function)
{
    if (!isComponentStateValid())
        return;

    auto* ncpFunction = dynamic_cast<NCPDebugNewtonFunctionInterface*>(&function);
    if (!ncpFunction)
    {
        msg_error() << "NCPDebugNewtonRaphsonSolver requires NCPDebugNewtonFunctionInterface.";
        d_lastFailureReason.setValue("configuration: residual function does not implement NCP interface");
        return;
    }

    start();
    d_lastAcceptedNewtonUpdates.setValue(0u);
    d_lastAcceptedAlpha.setValue(0_sreal);
    d_lastFailureReason.setValue(std::string());

    const bool keepLastAcceptedState = d_updateStateWhenDiverged.getValue();
    const SReal absoluteThreshold = d_absoluteResidualStoppingThreshold.getValue();
    const SReal relativeInitialThreshold = d_relativeInitialStoppingThreshold.getValue();
    const unsigned int maxNewtonIterations = d_maxNbIterationsNewton.getValue();
    const unsigned int maxLineSearchTrials = std::max(d_maxNbIterationsLineSearch.getValue(), 1u);
    const SReal reductionFactor = d_lineSearchCoefficient.getValue();

    const bool lmEnabled = d_enableLMFallback.getValue();
    const SReal lmInitialDamping = d_lmInitialDamping.getValue();
    const unsigned int lmMaxRetries = std::max(d_lmMaxRetries.getValue(), 1u);
    const unsigned int lmStagnationIterations = std::max(d_lmStagnationIterations.getValue(), 1u);

    ncpFunction->storeSolveState();
    function.evaluateCurrentGuess();

    SReal squaredResidual = function.squaredNormLastEvaluation();
    NCPDebugResidualSummary summary = ncpFunction->currentNCPDebugSummary();
    const SReal initialSquaredResidual = squaredResidual;

    if (!finiteNonNegative(squaredResidual))
    {
        d_lastFailureReason.setValue("residual-evaluation: initial residual is non-finite");
        msg_error() << "[NCP FAIL] stage=residual-evaluation reason=non-finite-initial-residual R2="
                    << squaredResidual << ".";
        return;
    }

    if (!summary.valid)
    {
        d_lastFailureReason.setValue("residual-evaluation: invalid contact evaluation");
        msg_error() << "[NCP FAIL] stage=residual-evaluation reason=invalid-contact-evaluation"
                    << " contacts=" << summary.activeContacts << "/" << summary.pinnedContacts << "/" << summary.invalidContacts
                    << " M=" << summary.mechanicalResidualNorm
                    << " C=" << summary.complementarityResidualNorm << ".";
        return;
    }

    std::deque<SReal> acceptedMeritHistory;
    acceptedMeritHistory.push_back(0.5_sreal * squaredResidual);

    unsigned int retainedUpdates = 0;
    unsigned int poorNewtonStepCount = 0;
    bool forceLMNextIteration = false;

    for (unsigned int iteration = 0; iteration < maxNewtonIterations; ++iteration)
    {
        SCOPED_TIMER_VARNAME(stepTimer, "NCPMeritNewtonStep");

        const SReal baseSquaredResidual = squaredResidual;
        const SReal baseResidual = normFromSquared(baseSquaredResidual);
        const NCPDebugResidualSummary baseSummary = summary;

        const bool absoluteThresholdCriterion = convergedAtAbsoluteThreshold(baseSquaredResidual, absoluteThreshold);
        const bool relativeInitialThresholdCriterion = convergedAtRelativeInitialThreshold(
            baseSquaredResidual, initialSquaredResidual, relativeInitialThreshold);

        if (absoluteThresholdCriterion || relativeInitialThresholdCriterion)
        {
            if (absoluteThresholdCriterion)
            {
                static constexpr NewtonStatus convergedAbsolute("ConvergedAbsoluteResidual");
                d_status.setValue(convergedAbsolute);
            }
            else
            {
                static constexpr NewtonStatus convergedRelativeInitial("ConvergedResidualInitialRatio");
                d_status.setValue(convergedRelativeInitial);
            }

            d_lastFailureReason.setValue(std::string());

            if (f_printLog.getValue() && d_logIterationSummary.getValue())
            {
                msg_info() << "[NCP CONVERGED] iterations=" << retainedUpdates
                           << " R=" << baseResidual
                           << " M=" << baseSummary.mechanicalResidualNorm
                           << " C=" << baseSummary.complementarityResidualNorm
                           << " criterion=" << (absoluteThresholdCriterion ? "absolute" : "relative-initial");
            }

            sofa::helper::AdvancedTimer::valSet("nb_iterations", retainedUpdates);
            sofa::helper::AdvancedTimer::valSet("residual", baseResidual);
            return;
        }

        if (f_printLog.getValue() && d_logIterationSummary.getValue())
        {
            msg_info() << "[NCP BASE " << retainedUpdates << "]"
                       << " R=" << baseResidual
                       << " M=" << baseSummary.mechanicalResidualNorm
                       << " C=" << baseSummary.complementarityResidualNorm
                       << " contacts=" << baseSummary.activeContacts << "/"
                       << baseSummary.pinnedContacts << "/" << baseSummary.invalidContacts
                       << " minGap=" << baseSummary.minimumActiveGap
                       << " penetration=" << baseSummary.maximumPenetration
                       << " lambda=[" << baseSummary.minimumLambda << "," << baseSummary.maximumLambda << "]";
        }

        ncpFunction->storeNewtonState();
        function.computeGradientFromCurrentGuess();

        const bool startWithLM = lmEnabled && forceLMNextIteration;
        const unsigned int maxDirectionAttempts = startWithLM
            ? lmMaxRetries + 1u  // LM rescue first; if all LM attempts fail, still allow Newton.
            : 1u + (lmEnabled ? lmMaxRetries : 0u);

        bool accepted = false;
        bool acceptedWithLM = false;
        bool acceptedWithGradient = false;
        SReal acceptedAlpha = 0_sreal;
        SReal acceptedRatio = std::numeric_limits<SReal>::infinity();

        bool anyValidTrial = false;
        SReal finalBestAlpha = 0_sreal;
        SReal finalBestRatio = std::numeric_limits<SReal>::infinity();
        SReal finalBestQM = std::numeric_limits<SReal>::infinity();
        SReal finalBestQC = std::numeric_limits<SReal>::infinity();
        SReal finalBestTrialResidual = std::numeric_limits<SReal>::infinity();

        unsigned int lmAttempt = 0u;

        for (unsigned int directionAttempt = 0; directionAttempt < maxDirectionAttempts && !accepted; ++directionAttempt)
        {
            const bool usingLM = startWithLM
                ? directionAttempt < lmMaxRetries
                : directionAttempt > 0u;
            SReal currentMu = 0_sreal;

            ncpFunction->restoreNewtonState();
            function.evaluateCurrentGuess();

            if (usingLM)
            {
                currentMu = lmInitialDamping * std::pow(10_sreal, static_cast<SReal>(lmAttempt));
                ++lmAttempt;

                const char* trigger = startWithLM ? "stagnation" : "line-search";
                if (f_printLog.getValue() && d_logIterationSummary.getValue())
                {
                    msg_warning() << "[NCP LM] trigger=" << trigger
                                  << " mu=" << currentMu
                                  << " attempt=" << lmAttempt << "/" << lmMaxRetries;
                }

                if (!ncpFunction->solveLevenbergMarquardt(currentMu))
                {
                    msg_warning() << "[NCP LM RETRY] mu=" << currentMu << " reason=linear-solve";
                    continue;
                }
            }
            else
            {
                function.solveLinearEquation();
            }

            const SReal squaredCorrection = function.squaredNormDx();
            if (!ncpFunction->correctionIsFinite() || !finiteNonNegative(squaredCorrection))
            {
                if (!usingLM && lmEnabled)
                    continue;

                if (usingLM && lmAttempt < lmMaxRetries)
                {
                    msg_warning() << "[NCP LM RETRY] mu=" << currentMu << " reason=non-finite-correction";
                    continue;
                }

                break;
            }

            SReal meritSlope = std::numeric_limits<SReal>::quiet_NaN();
            bool usedGradientFallback = false;

            if (!ncpFunction->selectMeritSearchDirection(
                    descentRho,
                    descentExponent,
                    meritSlope,
                    usedGradientFallback)
                || !std::isfinite(static_cast<double>(meritSlope))
                || !(meritSlope < 0_sreal))
            {
                if (!usingLM && lmEnabled)
                    continue;

                if (usingLM && lmAttempt < lmMaxRetries)
                {
                    msg_warning() << "[NCP LM RETRY] mu=" << currentMu << " reason=no-descent";
                    continue;
                }

                break;
            }

            const auto direction = ncpFunction->currentDirectionSummary();
            const char* directionType = usedGradientFallback ? "gradient" : (usingLM ? "lm" : "newton");

            if (f_printLog.getValue() && d_logIterationSummary.getValue())
            {
                msg_info() << "[NCP DIR " << retainedUpdates << "]"
                           << " type=" << directionType
                           << " |dz|=" << normFromSquared(function.squaredNormDx())
                           << " dPsi=" << meritSlope
                           << " |dx|=" << direction.translationNorm
                           << " |dtheta|=" << direction.rotationNorm
                           << " |dlambda|=" << direction.lambdaNorm
                           << " maxDx=" << direction.maximumNodeTranslation
                           << " maxDtheta=" << direction.maximumNodeRotation
                           << " maxDlambda=" << direction.maximumAbsLambda;
            }

            ncpFunction->beginFiniteDifferenceCheck();

            const SReal baseMerit = 0.5_sreal * baseSquaredResidual;
            SReal referenceMerit = baseMerit;

            bool sawValidTrial = false;
            SReal bestAlpha = 0_sreal;
            SReal bestRatio = std::numeric_limits<SReal>::infinity();
            SReal bestQM = std::numeric_limits<SReal>::infinity();
            SReal bestQC = std::numeric_limits<SReal>::infinity();
            SReal bestTrialResidual = std::numeric_limits<SReal>::infinity();
            SReal alpha = 1_sreal;

            for (unsigned int trial = 0; trial < maxLineSearchTrials; ++trial)
            {
                ncpFunction->setTrialFromNewtonState(alpha);
                function.evaluateCurrentGuess();

                const SReal trialSquaredResidual = function.squaredNormLastEvaluation();
                const NCPDebugResidualSummary trialSummary = ncpFunction->currentNCPDebugSummary();
                const bool validTrial = finiteNonNegative(trialSquaredResidual) && trialSummary.valid;
                const SReal trialResidual = validTrial
                    ? normFromSquared(trialSquaredResidual)
                    : std::numeric_limits<SReal>::infinity();

                const SReal qR = normRatio(trialResidual, baseResidual);
                const SReal qM = normRatio(trialSummary.mechanicalResidualNorm, baseSummary.mechanicalResidualNorm);
                const SReal qC = normRatio(trialSummary.complementarityResidualNorm, baseSummary.complementarityResidualNorm);

                ncpFunction->evaluateFiniteDifferenceTrial(alpha, retainedUpdates, trial);

                const SReal fdMeritSlope = validTrial && alpha > 0_sreal
                    ? (trialSquaredResidual - baseSquaredResidual) / (2_sreal * alpha)
                    : std::numeric_limits<SReal>::quiet_NaN();

                const SReal fdSlopeRatio = std::abs(meritSlope) > std::numeric_limits<SReal>::epsilon()
                    ? fdMeritSlope / meritSlope
                    : std::numeric_limits<SReal>::quiet_NaN();

                const SReal mechanicalGrowthFactor = 1.2_sreal;
                const SReal complementarityGrowthFactor = 1.2_sreal;
                const SReal mechanicalAbsoluteGuard = 1.0_sreal;
                const SReal complementarityAbsoluteGuard = 1.0_sreal;

                const SReal requiredMerit = referenceMerit + armijoCoefficient * alpha * meritSlope;
                const bool armijoAccepted = validTrial && 0.5_sreal * trialSquaredResidual < requiredMerit;
                const bool mechanicalAccepted = trialSummary.mechanicalResidualNorm
                    <= std::max(mechanicalAbsoluteGuard, mechanicalGrowthFactor * baseSummary.mechanicalResidualNorm);
                const bool complementarityAccepted = trialSummary.complementarityResidualNorm
                    <= std::max(complementarityAbsoluteGuard, complementarityGrowthFactor * baseSummary.complementarityResidualNorm);

                const bool trialAccepted = armijoAccepted;

                if (validTrial)
                {
                    sawValidTrial = true;
                    anyValidTrial = true;

                    if (qR < bestRatio)
                    {
                        bestAlpha = alpha;
                        bestRatio = qR;
                        bestQM = trialSummary.mechanicalResidualNorm;
                        bestQC = trialSummary.complementarityResidualNorm;
                        bestTrialResidual = trialResidual;
                    }

                    if (qR < finalBestRatio)
                    {
                        finalBestAlpha = alpha;
                        finalBestRatio = qR;
                        finalBestQM = trialSummary.mechanicalResidualNorm;
                        finalBestQC = trialSummary.complementarityResidualNorm;
                        finalBestTrialResidual = trialResidual;
                    }
                }

                if (f_printLog.getValue() && d_logLineSearchTrials.getValue() && validTrial)
                {
                    msg_info() << "[NCP LS " << retainedUpdates << "." << trial << "]"
                               << " type=" << directionType
                               << " alpha=" << alpha
                               << " qR=" << qR
                               << " qM=" << qM
                               << " qC=" << qC
                               << " fdSlope=" << fdMeritSlope
                               << " predictedSlope=" << meritSlope
                               << " fdSlopeRatio=" << fdSlopeRatio
                               << " armijo=" << armijoAccepted
                               << " mechanicalAccepted=" << mechanicalAccepted
                               << " complementarityAccepted=" << complementarityAccepted
                               << " stepMaxDx=" << alpha * direction.maximumNodeTranslation
                               << " stepMaxDtheta=" << alpha * direction.maximumNodeRotation
                               << " stepMaxDlambda=" << alpha * direction.maximumAbsLambda;
                }

                if (trialAccepted)
                {
                    accepted = true;
                    acceptedWithLM = usingLM;
                    acceptedWithGradient = usedGradientFallback;
                    acceptedAlpha = alpha;
                    acceptedRatio = qR;
                    squaredResidual = trialSquaredResidual;
                    summary = trialSummary;

                    if (f_printLog.getValue() && d_logIterationSummary.getValue())
                    {
                        msg_info() << "[NCP ACCEPT " << retainedUpdates << "]"
                                   << " type=" << directionType
                                   << " alpha=" << alpha
                                   << " qR=" << qR
                                   << " qM=" << qM
                                   << " qC=" << qC;
                    }
                    break;
                }

                alpha *= reductionFactor;
            }

            if (accepted)
                break;

            ncpFunction->restoreNewtonState();
            function.evaluateCurrentGuess();

            if (usingLM)
            {
                if (lmAttempt < lmMaxRetries)
                {
                    msg_warning_when(d_warnWhenLineSearchFails.getValue())
                        << "[NCP LM RETRY] mu=" << currentMu
                        << " reason=line-search"
                        << " bestAlpha=" << bestAlpha
                        << " bestqR=" << bestRatio;
                }
            }
        }

        if (accepted)
        {
            ++retainedUpdates;
            acceptedMeritHistory.push_back(0.5_sreal * squaredResidual);

            const unsigned int nonMonotoneWindow = std::max(d_nonMonotoneWindow.getValue(), 1u);
            while (acceptedMeritHistory.size() > nonMonotoneWindow)
                acceptedMeritHistory.pop_front();

            d_lastAcceptedNewtonUpdates.setValue(retainedUpdates);
            d_lastAcceptedAlpha.setValue(acceptedAlpha);

            if (acceptedWithLM || acceptedWithGradient)
            {
                poorNewtonStepCount = 0u;
                forceLMNextIteration = false;
            }
            else
            {
                const bool poorNewtonStep = acceptedAlpha < lmStagnationAlphaThreshold
                    && acceptedRatio > lmStagnationResidualRatio;

                poorNewtonStepCount = poorNewtonStep ? poorNewtonStepCount + 1u : 0u;
                forceLMNextIteration = lmEnabled && poorNewtonStepCount >= lmStagnationIterations;

                if (forceLMNextIteration && f_printLog.getValue() && d_logIterationSummary.getValue())
                {
                    msg_warning() << "[NCP LM ARM] reason=stagnation"
                                  << " count=" << poorNewtonStepCount
                                  << " alpha=" << acceptedAlpha
                                  << " qR=" << acceptedRatio;
                }
            }

            continue;
        }

        ncpFunction->restoreNewtonState();
        function.evaluateCurrentGuess();
        squaredResidual = function.squaredNormLastEvaluation();
        summary = ncpFunction->currentNCPDebugSummary();

        if (!keepLastAcceptedState)
        {
            ncpFunction->restoreSolveState();
            function.evaluateCurrentGuess();
            squaredResidual = function.squaredNormLastEvaluation();
            summary = ncpFunction->currentNCPDebugSummary();
            retainedUpdates = 0;
        }

        d_lastAcceptedNewtonUpdates.setValue(retainedUpdates);
        d_lastAcceptedAlpha.setValue(0_sreal);

        static constexpr NewtonStatus divergedLineSearch("DivergedLineSearch");
        d_status.setValue(divergedLineSearch);
        d_lastFailureReason.setValue("line-search: Newton and LM fallback failed");

        msg_warning_when(d_warnWhenLineSearchFails.getValue())
            << "[NCP FAIL] stage=line-search reason=newton-and-lm-failed"
            << " bestAlpha=" << finalBestAlpha
            << " bestqR=" << finalBestRatio
            << " bestResidual=" << finalBestTrialResidual
            << " Trial M=" << finalBestQM
            << " Trial C=" << finalBestQC
            << " Base R=" << baseResidual
            << " Base M=" << baseSummary.mechanicalResidualNorm
            << " Base C=" << baseSummary.complementarityResidualNorm
            << " retainedUpdates=" << retainedUpdates << ".";

        sofa::helper::AdvancedTimer::valSet("nb_iterations", retainedUpdates);
        sofa::helper::AdvancedTimer::valSet("residual", normFromSquared(squaredResidual));
        return;
    }

    if (!keepLastAcceptedState)
    {
        ncpFunction->restoreSolveState();
        function.evaluateCurrentGuess();
        squaredResidual = function.squaredNormLastEvaluation();
        retainedUpdates = 0;
        d_lastAcceptedNewtonUpdates.setValue(0u);
        d_lastAcceptedAlpha.setValue(0_sreal);
    }

    static constexpr NewtonStatus divergedMaxIterations("DivergedMaxIterations");
    d_status.setValue(divergedMaxIterations);
    d_lastFailureReason.setValue("max-iterations: nonlinear solve did not converge");

    msg_warning_when(d_warnWhenDiverge.getValue())
        << "[NCP FAIL] stage=max-iterations retainedUpdates=" << retainedUpdates
        << " R=" << normFromSquared(squaredResidual) << ".";

    sofa::helper::AdvancedTimer::valSet("nb_iterations", retainedUpdates);
    sofa::helper::AdvancedTimer::valSet("residual", normFromSquared(squaredResidual));
}

} // namespace sofa::ncp
