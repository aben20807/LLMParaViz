<template>
  <div class="communication-view">
    <h2>📡 Layer Communication Breakdown</h2>
    
    <p class="description">Shows collective operations between transformer layer i and i+1</p>

    <!-- Layer-wise Component Breakdown -->
    <div class="layer-breakdown">
      <!-- Layer i header -->
      <div class="layer-header">
        <span class="layer-badge">Layer i</span>
        <span class="layer-desc">Forward Pass Components</span>
      </div>

      <!-- Component Timeline -->
      <div class="component-timeline">
        <!-- Attention Block -->
        <div class="component-block attention-block">
          <div class="block-header">
            <span class="block-icon">🔍</span>
            <span class="block-title">Attention</span>
          </div>
          
          <div class="block-stages">
            <!-- QKV Projection -->
            <div class="stage">
              <div class="stage-name">QKV Projection</div>
              <div class="stage-comm" v-if="config.fsdp > 1">
                <span class="comm-badge fsdp">AllGather</span>
                <span class="comm-label">FSDP weights</span>
              </div>
              <div class="stage-comm none" v-else>
                <span class="comm-badge none">—</span>
              </div>
            </div>

            <!-- Attention Compute with Sequence Parallel -->
            <div class="stage" v-if="hasAttentionDP">
              <div class="stage-name">Pre-Attention</div>
              <div class="stage-comm">
                <span class="comm-badge adp">All2All</span>
                <span class="comm-label">Attention DP (scatter seq, gather heads)</span>
              </div>
            </div>

            <!-- Attention Compute -->
            <div class="stage">
              <div class="stage-name">Attention Compute</div>
              <div class="stage-comm" v-if="config.cp > 1">
                <span class="comm-badge cp">Ring</span>
                <span class="comm-label">Context Parallel KV exchange</span>
              </div>
              <div class="stage-comm none" v-else>
                <span class="comm-badge none">—</span>
                <span class="comm-label">Local compute</span>
              </div>
            </div>

            <!-- Post-Attention with Sequence Parallel -->
            <div class="stage" v-if="hasAttentionDP">
              <div class="stage-name">Post-Attention</div>
              <div class="stage-comm">
                <span class="comm-badge adp">All2All</span>
                <span class="comm-label">Attention DP (gather seq, scatter heads)</span>
              </div>
            </div>

            <!-- Output Projection -->
            <div class="stage">
              <div class="stage-name">Output Projection</div>
              <div class="stage-comm" v-if="config.attn_tp > 1">
                <span class="comm-badge attn_tp">AllReduce</span>
                <span class="comm-label">Attention TP</span>
              </div>
              <div class="stage-comm" v-else-if="config.fsdp > 1">
                <span class="comm-badge fsdp">ReduceScatter</span>
                <span class="comm-label">FSDP grads</span>
              </div>
              <div class="stage-comm none" v-else>
                <span class="comm-badge none">—</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Residual Add -->
        <div class="connector">
          <div class="connector-line"></div>
          <span class="connector-label">+ Residual</span>
        </div>

        <!-- FFN / MoE Block -->
        <div class="component-block ffn-block">
          <div class="block-header">
            <span class="block-icon">{{ hasMoE ? '🔀' : '⚡' }}</span>
            <span class="block-title">{{ hasMoE ? 'MoE FFN' : 'FFN' }}</span>
          </div>
          
          <div class="block-stages">
            <!-- Router (MoE only) -->
            <div class="stage" v-if="hasMoE">
              <div class="stage-name">Router</div>
              <div class="stage-comm none">
                <span class="comm-badge none">—</span>
                <span class="comm-label">Local routing decision</span>
              </div>
            </div>

            <!-- Token Dispatch (MoE only) -->
            <div class="stage" v-if="hasMoE && config.ep > 1">
              <div class="stage-name">Token Dispatch</div>
              <div class="stage-comm">
                <span class="comm-badge ep">All2All</span>
                <span class="comm-label">EP: send tokens to experts</span>
              </div>
            </div>

            <!-- FFN Up Projection -->
            <div class="stage">
              <div class="stage-name">{{ hasMoE ? 'Expert' : 'FFN' }} Up Proj</div>
              <div class="stage-comm" v-if="config.fsdp > 1">
                <span class="comm-badge fsdp">AllGather</span>
                <span class="comm-label">FSDP weights</span>
              </div>
              <div class="stage-comm none" v-else>
                <span class="comm-badge none">—</span>
              </div>
            </div>

            <!-- FFN Down Projection -->
            <div class="stage">
              <div class="stage-name">{{ hasMoE ? 'Expert' : 'FFN' }} Down Proj</div>
              <div class="stage-comm" v-if="config.expert_tp > 1">
                <span class="comm-badge expert_tp">AllReduce</span>
                <span class="comm-label">Expert TP</span>
              </div>
              <div class="stage-comm" v-else-if="config.fsdp > 1">
                <span class="comm-badge fsdp">ReduceScatter</span>
                <span class="comm-label">FSDP grads</span>
              </div>
              <div class="stage-comm none" v-else>
                <span class="comm-badge none">—</span>
              </div>
            </div>

            <!-- Token Combine (MoE only) -->
            <div class="stage" v-if="hasMoE && config.ep > 1">
              <div class="stage-name">Token Combine</div>
              <div class="stage-comm">
                <span class="comm-badge ep">All2All</span>
                <span class="comm-label">EP: gather expert outputs</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Residual Add -->
        <div class="connector">
          <div class="connector-line"></div>
          <span class="connector-label">+ Residual</span>
        </div>
      </div>

      <!-- Layer i+1 header -->
      <div class="layer-header next-layer">
        <span class="layer-badge">Layer i+1</span>
        <span class="layer-desc" v-if="config.pp > 1">(may be on different PP stage)</span>
      </div>

      <!-- Pipeline Parallel Communication -->
      <div class="pp-section" v-if="config.pp > 1">
        <div class="pp-comm">
          <span class="comm-badge pp">P2P Send/Recv</span>
          <span class="comm-label">Pipeline Parallel: activations between stages</span>
        </div>
      </div>
    </div>

    <!-- Gradient Sync Section -->
    <div class="gradient-section">
      <div class="section-header">
        <span class="section-icon">🔄</span>
        <span class="section-title">Backward Pass Gradient Sync</span>
      </div>
      <div class="gradient-comms">
        <div class="grad-item" v-if="config.dp > 1">
          <span class="comm-badge dp">AllReduce</span>
          <span class="comm-label">Data Parallel gradient averaging</span>
        </div>
        <div class="grad-item" v-if="config.fsdp > 1">
          <span class="comm-badge fsdp">ReduceScatter</span>
          <span class="comm-label">FSDP gradient sharding</span>
        </div>
        <div class="grad-item" v-if="config.dp <= 1 && config.fsdp <= 1">
          <span class="comm-badge none">—</span>
          <span class="comm-label">No gradient sync (single replica)</span>
        </div>
      </div>
    </div>

    <!-- Summary Table -->
    <div class="summary-section">
      <div class="section-header">
        <span class="section-icon">📊</span>
        <span class="section-title">Active Collectives Summary</span>
      </div>
      <div class="summary-table">
        <div 
          v-for="comm in activeSummary" 
          :key="comm.key"
          class="summary-row"
        >
          <span class="comm-badge" :class="comm.key">{{ comm.operation }}</span>
          <span class="summary-dim">{{ comm.label }}</span>
          <span class="summary-size">{{ comm.groupSize }} GPUs/group</span>
          <span class="summary-count">{{ comm.count }} group(s)</span>
        </div>
        <div v-if="activeSummary.length === 0" class="summary-row empty">
          <span class="comm-label">No active parallelism (single GPU)</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  config: {
    type: Object,
    required: true
  }
})

const hasAttentionDP = computed(() => (props.config.ulysses_dp || 1) > 1)
const hasMoE = computed(() => props.config.ep > 1 || props.config.expert_tp > 1)

const totalGPUs = computed(() => {
  const c = props.config
  if (!c) return 1
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  return base * c.expert_tp * c.cp * c.fsdp * c.dp * c.pp
})

const activeSummary = computed(() => {
  const c = props.config
  if (!c) return []
  
  const items = []
  const ulysses = c.ulysses_dp || 1
  const total = totalGPUs.value
  
  if (c.attn_tp > 1) {
    items.push({ 
      key: 'attn_tp', 
      label: 'Attention TP', 
      operation: 'AllReduce', 
      groupSize: c.attn_tp,
      count: total / c.attn_tp
    })
  }
  if (ulysses > 1) {
    items.push({ 
      key: 'adp', 
      label: 'Attention DP', 
      operation: 'All2All', 
      groupSize: ulysses,
      count: total / ulysses
    })
  }
  if (c.expert_tp > 1) {
    items.push({ 
      key: 'expert_tp', 
      label: 'Expert TP', 
      operation: 'AllReduce', 
      groupSize: c.expert_tp,
      count: total / c.expert_tp
    })
  }
  if (c.ep > 1) {
    items.push({ 
      key: 'ep', 
      label: 'Expert Parallel', 
      operation: 'All2All', 
      groupSize: c.ep,
      count: total / c.ep
    })
  }
  if (c.cp > 1) {
    items.push({ 
      key: 'cp', 
      label: 'Context Parallel', 
      operation: 'Ring', 
      groupSize: c.cp,
      count: total / c.cp
    })
  }
  if (c.fsdp > 1) {
    items.push({ 
      key: 'fsdp', 
      label: 'FSDP', 
      operation: 'AllGather/RS', 
      groupSize: c.fsdp,
      count: total / c.fsdp
    })
  }
  if (c.dp > 1) {
    items.push({ 
      key: 'dp', 
      label: 'Data Parallel', 
      operation: 'AllReduce', 
      groupSize: c.dp,
      count: total / c.dp
    })
  }
  if (c.pp > 1) {
    items.push({ 
      key: 'pp', 
      label: 'Pipeline Parallel', 
      operation: 'P2P', 
      groupSize: c.pp,
      count: total / c.pp
    })
  }
  
  return items
})
</script>

<style scoped>
.communication-view {
  background-color: var(--panel-bg);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.communication-view h2 {
  margin: 0 0 8px 0;
  font-size: var(--font-xl);
  font-weight: 700;
  color: var(--text-primary);
}

.description {
  margin: 0 0 20px 0;
  font-size: var(--font-sm);
  color: var(--text-secondary);
  line-height: var(--leading-relaxed);
}

/* Layer Breakdown */
.layer-breakdown {
  background-color: var(--control-bg);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
}

.layer-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}

.layer-header.next-layer {
  margin-top: 10px;
  margin-bottom: 0;
}

.layer-badge {
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
  color: white;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: var(--font-xs);
  font-weight: 700;
}

.layer-desc {
  font-size: var(--font-sm);
  color: var(--text-secondary);
}

/* Component Timeline */
.component-timeline {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.component-block {
  background-color: var(--panel-bg);
  border-radius: 10px;
  padding: 14px;
  border-left: 4px solid;
}

.attention-block {
  border-left-color: #7c3aed;
}

.ffn-block {
  border-left-color: #ea580c;
}

.block-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.block-icon {
  font-size: 1.125rem;
}

.block-title {
  font-weight: 700;
  font-size: var(--font-base);
  color: var(--text-primary);
}

.block-stages {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stage {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background-color: var(--control-bg);
  border-radius: 8px;
  font-size: var(--font-sm);
  flex-wrap: wrap;
  gap: 8px;
}

.stage-name {
  font-weight: 600;
  color: var(--text-primary);
  min-width: 130px;
}

.stage-comm {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  justify-content: flex-end;
}

.stage-comm.none {
  opacity: 0.6;
}

/* Connector */
.connector {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 14px;
}

.connector-line {
  flex: 1;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--border-color), transparent);
}

.connector-label {
  font-size: var(--font-xs);
  color: var(--text-muted);
  white-space: nowrap;
}

/* Communication Badges */
.comm-badge {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: var(--font-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.comm-badge.attn_tp {
  background-color: #dc2626;
  color: white;
}

.comm-badge.adp {
  background-color: #b91c1c;
  color: white;
}

.comm-badge.expert_tp {
  background-color: #ea580c;
  color: white;
}

.comm-badge.ep {
  background-color: #059669;
  color: white;
}

.comm-badge.cp {
  background-color: #7c3aed;
  color: white;
}

.comm-badge.fsdp {
  background-color: #db2777;
  color: white;
}

.comm-badge.dp {
  background-color: #2563eb;
  color: white;
}

.comm-badge.pp {
  background-color: #d97706;
  color: white;
}

.comm-badge.none {
  background-color: #6b7280;
  color: white;
}

.comm-label {
  font-size: var(--font-xs);
  color: var(--text-secondary);
  line-height: var(--leading-normal);
}

/* PP Section */
.pp-section {
  margin-top: 14px;
  padding: 12px;
  background-color: rgba(217, 119, 6, 0.1);
  border-radius: 8px;
  border: 1px dashed #d97706;
}

.pp-comm {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}

/* Gradient Section */
.gradient-section {
  background-color: var(--control-bg);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 20px;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.section-icon {
  font-size: 1.125rem;
}

.section-title {
  font-weight: 700;
  font-size: var(--font-base);
  color: var(--text-primary);
}

.gradient-comms {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.grad-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background-color: var(--panel-bg);
  border-radius: 8px;
}

/* Summary Section */
.summary-section {
  background-color: var(--control-bg);
  border-radius: 10px;
  padding: 16px;
}

.summary-table {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.summary-row {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 10px 12px;
  background-color: var(--panel-bg);
  border-radius: 8px;
  font-size: var(--font-sm);
  flex-wrap: wrap;
}

.summary-row.empty {
  justify-content: center;
  opacity: 0.6;
}

.summary-dim {
  font-weight: 600;
  color: var(--text-primary);
  min-width: 130px;
}

.summary-size {
  color: var(--text-secondary);
  min-width: 110px;
}

.summary-count {
  color: var(--text-muted);
  font-size: var(--font-xs);
}
</style>
