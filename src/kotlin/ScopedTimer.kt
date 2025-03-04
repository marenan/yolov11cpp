package com.yolov11kotlin

import android.os.SystemClock
import android.util.Log

/**
 * Utility class for measuring execution time of code blocks.
 * Only logs times when TIMING_MODE is enabled in the BuildConfig.
 */
class ScopedTimer(private val name: String) {
    private val startTime: Long = SystemClock.elapsedRealtime()
    private var stopped = false

    /**
     * Stops the timer and logs the elapsed time.
     */
    fun stop() {
        if (stopped) return
        stopped = true
        
        if (BuildConfig.TIMING_MODE) {
            val endTime = SystemClock.elapsedRealtime()
            val duration = endTime - startTime
            Log.d("ScopedTimer", "$name took $duration milliseconds.")
        }
    }

    /**
     * Automatically stops the timer when the object is garbage collected.
     */
    protected fun finalize() {
        if (!stopped) {
            stop()
        }
    }
}
