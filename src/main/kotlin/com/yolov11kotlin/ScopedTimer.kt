package com.yolov11kotlin

import kotlin.system.measureTimeMillis

/**
 * Utility class for measuring execution time of code blocks
 */
class ScopedTimer(private val label: String) {
    private val startTime = System.currentTimeMillis()
    
    /**
     * Records the elapsed time and prints it when the timer is closed
     */
    fun close() {
        val elapsed = System.currentTimeMillis() - startTime
        if (Config.TIMING_MODE) {
            println("$label took $elapsed ms")
        }
    }
    
    companion object {
        /**
         * Execute a block of code and measure its execution time
         */
        inline fun <T> measure(label: String, block: () -> T): T {
            var result: T
            val elapsed = measureTimeMillis {
                result = block()
            }
            if (Config.TIMING_MODE) {
                println("$label took $elapsed ms")
            }
            return result
        }
    }
}
