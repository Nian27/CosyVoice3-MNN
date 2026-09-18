package com.cosyvoice.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CosyVoiceLlmAttemptPlanTest {

    @Test
    fun openclRouteRunsOpenclFiveTimesThenCpuTwice() {
        assertEquals(
            listOf("opencl", "opencl", "opencl", "opencl", "opencl", "cpu", "cpu"),
            CosyVoiceLlmAttemptPlan.backends("opencl")
        )
    }

    @Test
    fun hexagonRouteKeepsCpuAsLastResort() {
        assertEquals(
            listOf("hexagon", "hexagon", "hexagon", "hexagon", "hexagon", "cpu", "cpu"),
            CosyVoiceLlmAttemptPlan.backends("hexagon")
        )
    }

    // 入口后端本来就是 cpu 时，兜底块会把同一份 CPU 配置再跑一遍（每次约 70 s）——
    // 现在只有 2 次，且第 2 次也过质量门。
    @Test
    fun cpuRouteDoesNotReRunIdenticalCpuJobMoreThanTwice() {
        assertEquals(listOf("cpu", "cpu"), CosyVoiceLlmAttemptPlan.backends("cpu"))
    }

    // 以下向量都是设备实测原始产物（SM8850，2026-09-18）。
    // 见 docs/DECISIONS.md 的 ADR-053 增补 NPU-R5 / R7 / R9。

    // 同句 63 字、短输出下的干净产物（249 token、末位 EOS、无 >= 8 的连续段）。
    @Test
    fun capturedOpenclRunPassesTheGate() {
        val tokens = readFixture("llm-tokens/opencl-clean-249.csv")

        assertTrue("原始 249 个 token", tokens.size == 249)
        assertTrue("末位是 EOS", tokens.last() == 158486)
        assertNull(CosyVoiceLlmOutputQuality.collapseReason(tokens))
    }

    // 长输出的退化：500 token、无 EOS、尾段连续 266 个相同 token（撞满 maxTokens）。
    @Test
    fun capturedRunawayIsRejectedForMissingEos() {
        val tokens = readFixture("llm-tokens/cpu-collapse-500.csv")

        assertTrue("原始 500 个 token", tokens.size == 500)
        assertTrue("没有 EOS", tokens.none { it == 158486 })
        val reason = CosyVoiceLlmOutputQuality.collapseReason(tokens)
        assertNotNull(reason)
        assertTrue(reason!!, reason.contains("未生成 EOS"))
        // 重复段起点（234）与长度（266）都写进拒因，供事后判断是否为序列长度边界。
        assertTrue(reason, reason.contains("len266@[234-499]"))
    }

    // 反向回归：有 EOS、370 token、内部最长重复段 10 —— 旧判据（longestRun >= 8）会误杀它，
    // 于是 7 次尝试全部被拒、整条链直接报错。新判据必须放行。
    @Test
    fun capturedLongOutputWithEosAndShortRunIsAccepted() {
        val tokens = readFixture("llm-tokens/opencl-eos-run10-370.csv")

        assertTrue("原始 370 个 token", tokens.size == 370)
        assertTrue("末位是 EOS", tokens.last() == 158486)
        assertNull(CosyVoiceLlmOutputQuality.collapseReason(tokens))
    }

    // 第 3 条判据：即使有 EOS，单段连续重复 >= 64 仍算退化。
    @Test
    fun singleRunAtTheLimitIsRejectedEvenWithEos() {
        val tokens = (1..120).toMutableList() + List(64) { 7777 } + listOf(158486)

        val reason = CosyVoiceLlmOutputQuality.collapseReason(tokens)
        assertNotNull(reason)
        assertTrue(reason!!, reason.contains("单段连续重复 64"))
    }

    @Test
    fun singleRunJustBelowTheLimitPasses() {
        val tokens = (1..120).toMutableList() + List(63) { 7777 } + listOf(158486)

        assertNull(CosyVoiceLlmOutputQuality.collapseReason(tokens))
    }

    // R9 上线后的首次真机验证：同句 63 字、首个尝试即被接受的那一次
    // （380 token、末位 EOS、unique=236、最长重复段 17@[231-247]）。
    @Test
    fun capturedAcceptedProductionRunPassesTheGate() {
        val tokens = readFixture("llm-tokens/opencl-eos-run17-380.csv")

        assertTrue("原始 380 个 token", tokens.size == 380)
        assertTrue("末位是 EOS", tokens.last() == 158486)
        assertNull(CosyVoiceLlmOutputQuality.collapseReason(tokens))
    }

    // R10 的关键向量：长句（63 字 / inputTokens=196）走 hexagon，首试即通过的那一次
    // （409 token、末位 EOS、unique=250、longestRun=15@[76-90]）。旧判据同样会拒掉它。
    @Test
    fun capturedHexagonLongTextRunPassesTheGate() {
        val tokens = readFixture("llm-tokens/hexagon-eos-run15-409.csv")

        assertTrue("原始 409 个 token", tokens.size == 409)
        assertTrue("末位是 EOS", tokens.last() == 158486)
        assertNull(CosyVoiceLlmOutputQuality.collapseReason(tokens))
    }

    // ---- 尾部空转修复（R11）----

    // 同一个形态在三个后端上都出现过：填满 500、没有 EOS、尾部一长段同一个 token 一直顶到 499。
    @Test
    fun tailRunawayIsRepairedByTrimmingTheTrailingRun() {
        val cpu = readFixture("llm-tokens/cpu-collapse-500.csv")
        val hex = readFixture("llm-tokens/hexagon-tail-runaway-500.csv")

        val cpuRepair = CosyVoiceLlmTailRepair.repair(cpu)
        assertNotNull(cpuRepair)
        assertEquals(234, cpuRepair!!.tokens.size)
        assertEquals(266, cpuRepair.trimmedRun)

        val hexRepair = CosyVoiceLlmTailRepair.repair(hex)
        assertNotNull(hexRepair)
        assertEquals(370, hexRepair!!.tokens.size)
        assertEquals(130, hexRepair.trimmedRun)
    }

    @Test
    fun tailRepairLeavesProperlyTerminatedRunsAlone() {
        assertNull(CosyVoiceLlmTailRepair.repair(readFixture("llm-tokens/opencl-clean-249.csv")))
        assertNull(CosyVoiceLlmTailRepair.repair(readFixture("llm-tokens/hexagon-eos-run15-409.csv")))
    }

    @Test
    fun tailRepairRequiresALongTailRunAndEnoughKeptTokens() {
        // 尾部只有 31 个重复：不动。
        val shortTail = (1..200).toMutableList() + List(31) { 4242 }
        assertNull(CosyVoiceLlmTailRepair.repair(shortTail))

        // 尾部重复够长，但保留段不足 32：不动。
        val tinyHead = List(20) { it + 1 } + List(60) { 4242 }
        assertNull(CosyVoiceLlmTailRepair.repair(tinyHead))
    }

    // Native 的 speech-tokens-N.csv 约定：token - 151924，逗号分隔，末尾换行。
    @Test
    fun speechTokenLineFollowsTheNativeLayout() {
        val line = CosyVoiceLlmTailRepair.speechTokenLine(listOf(151926, 152000, 999))

        assertEquals("2,76\n", line)
    }

    private fun readFixture(path: String): List<Int> {
        val stream = checkNotNull(javaClass.classLoader?.getResourceAsStream(path)) {
            "缺少测试向量 $path"
        }
        return stream.bufferedReader().use { it.readText() }
            .split(',')
            .mapNotNull { it.trim().toIntOrNull() }
    }
}
