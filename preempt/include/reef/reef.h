#ifndef __REEF_REEF_H__
#define __REEF_REEF_H__

#include <cstddef>
#include <cstdint>

#define EXPORT_REEF_FUNC extern "C" __attribute__((visibility("default")))

typedef struct CUstream_st* CUstream; // forward declear cuda stream type

namespace reef
{
    enum RFresult
    {
        REEF_SUCCESS                = 0,
        REEF_ERROR_INVALID_VALUE    = 1,
        REEF_ERROR_NOT_FOUND        = 2,
        REEF_ERROR_ALREADY_EXIST    = 3,
    };

    //!
    //! \enum PreemptLevel
    //! \brief Controls what REEF will do when preemption happens.
    //!
    //! The higher the level, the faster the preemption.
    //!
    enum PreemptLevel
    {
        PreemptHostQueue    = 0, // REEF will: stop submitting tasks on the background thread.
        PreemptDeviceQueue  = 1, // REEF will: stop submitting tasks + set the preemption flag to clean vDeviceQueue.
        PreemptAll          = 2, // REEF will: stop submitting tasks + set the preemption flag + abort the running kernel (not supported yet)
    };

    struct RFconfig
    {
        size_t queueSize;       // The bigger, the less overhead when execution, the slower the preemption
        size_t batchSize;       // The bigger, the less overhead when execution, should not be bigger than queueSize
        int32_t taskTimeout;    // If no stream commands enqueue after taskTimeout, the task will be considered finished. Measured in microseconds (us). taskTimeout < 0 means disable timeout mechanism.
        PreemptLevel preemptLevel;
    };

    static const RFconfig defaultConfig
    {
        .queueSize = 64,
        .batchSize = 16,

#ifdef NO_SCHED
        .taskTimeout = -1,
#else
        .taskTimeout = 500,
#endif

#ifdef DEFAULT_PREEMPT_LEVEL_HOST_QUEUE
        .preemptLevel = PreemptHostQueue,
#else
        .preemptLevel = PreemptDeviceQueue,
#endif
    };

    //!
    //! \brief enable preemption capability on a cuda stream,
    //! should be call before calling preempt() and restore()
    //!
    //! \param cuStream The cuStream that should enable preemption capability.
    //! \param config The configuration that controls how REEF acts.
    //!        Default config will be used if omitted.
    //!
    //! \return REEF_SUCCESS if succeeded, others if failed
    //!
    EXPORT_REEF_FUNC RFresult enablePreemption(CUstream cuStream, RFconfig config=defaultConfig);

    //!
    //! \brief diable preemption capability on a cuda stream
    //!
    //! \param cuStream The cuStream that have already enabled preemption capability.
    //!
    //! \return REEF_SUCCESS if succeeded, others if failed
    //!
    EXPORT_REEF_FUNC RFresult disablePreemption(CUstream cuStream);
    
    //!
    //! \brief preempt a running cuda stream, all submitted tasks on this stream will be suspended.
    //!
    //! \param cuStream The cuStream that have already enabled preemption capability.
    //! \param synchronize Whether to wait until the stream is stopped and all tasks are suspended.
    //!
    //! \return REEF_SUCCESS if succeeded, others if failed
    //!
    EXPORT_REEF_FUNC RFresult preempt(CUstream cuStream, bool synchronize=false);

    //!
    //! \brief restore a preempted cuda stream, all submitted tasks on this stream will be resumed.
    //!
    //! \param cuStream The cuStream that have already enabled preemption capability.
    //!
    //! \return REEF_SUCCESS if succeeded, others if failed
    //!
    EXPORT_REEF_FUNC RFresult restore(CUstream cuStream);

} // namespace reef

#endif