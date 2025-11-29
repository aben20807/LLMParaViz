<template>
  <div class="info-panel">
    <h2>📚 Understanding Parallelism</h2>
    
    <div class="info-tabs">
      <button 
        v-for="tab in tabs" 
        :key="tab.key"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>
    
    <div class="info-content">
      <!-- Overview Tab -->
      <div v-if="activeTab === 'overview'" class="tab-content">
        <div class="overview-grid">
          <div class="overview-card">
            <h4>Current Configuration</h4>
            <div class="config-summary">
              <div class="config-formula">
                {{ configFormula }}
              </div>
              <div class="config-total">
                = <strong>{{ totalGPUs }}</strong> GPUs
              </div>
            </div>
          </div>
          
          <div class="overview-card">
            <h4>Memory per GPU</h4>
            <div class="memory-breakdown">
              <div class="memory-item" v-for="item in memoryEstimate" :key="item.label">
                <span class="memory-label">{{ item.label }}</span>
                <span class="memory-factor">{{ item.factor }}</span>
              </div>
            </div>
          </div>
          
          <div class="overview-card">
            <h4>Communication Patterns</h4>
            <div class="comm-list">
              <div v-for="comm in activeComms" :key="comm.type" class="comm-item">
                <span class="comm-type" :style="{ color: comm.color }">{{ comm.type }}</span>
                <span class="comm-op">{{ comm.operation }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Parallelism Types Tab -->
      <div v-if="activeTab === 'types'" class="tab-content">
        <div class="parallelism-cards">
          <div v-for="p in parallelismTypes" :key="p.key" class="parallelism-card">
            <div class="p-header" :style="{ borderColor: p.color }">
              <span class="p-abbrev" :style="{ background: p.color }">{{ p.abbrev }}</span>
              <span class="p-name">{{ p.name }}</span>
            </div>
            <p class="p-description">{{ p.description }}</p>
            <div class="p-details">
              <div class="p-detail">
                <strong>Shards:</strong> {{ p.shards }}
              </div>
              <div class="p-detail">
                <strong>Communication:</strong> {{ p.communication }}
              </div>
              <div class="p-detail">
                <strong>Use when:</strong> {{ p.useCase }}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Mesh Hierarchy Tab -->
      <div v-if="activeTab === 'mesh'" class="tab-content">
        <div class="mesh-explanation">
          <h4>Device Mesh Hierarchy</h4>
          <p>The parallelism dimensions are ordered <strong>[TP, EP, CP, FSDP, DP, PP]</strong> from closest (fastest) to furthest (slowest):</p>
          
          <div class="mesh-order">
            <div 
              v-for="(dim, index) in meshOrder" 
              :key="dim.key"
              class="mesh-dim"
              :style="{ borderLeftColor: dim.color }"
            >
              <span class="mesh-rank">{{ index + 1 }}</span>
              <div class="mesh-info">
                <span class="mesh-name" :style="{ color: dim.color }">{{ dim.name }}</span>
                <span class="mesh-desc">{{ dim.placement }}</span>
              </div>
            </div>
          </div>
          
          <div class="mesh-note">
            <strong>At Scale Example (32K+ GPUs):</strong><br>
            • <strong>TP=8</strong> within-node via NVSwitch<br>
            • <strong>EP=16</strong> across rail-optimized leaf switches for All2All<br>
            • <strong>CP×FS×DP=256</strong> fills a 32K GPU island<br>
            • <strong>PP=?</strong> across islands for pipeline stages<br>
            <em>Latency priority: CP > FSDP > DP</em>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  config: Object
})

const activeTab = ref('overview')

const tabs = [
  { key: 'overview', label: 'Overview' },
  { key: 'types', label: 'Parallelism Types' },
  { key: 'mesh', label: 'Mesh Hierarchy' }
]

const totalGPUs = computed(() => {
  const c = props.config
  // Attention TP and EP can share the same GPU dimension (orthogonal sharding)
  // Ulysses DP multiplies with attn_tp for effective attention parallelism
  // Expert TP further shards each expert, so it multiplies
  // Formula: max(attn_tp × ulysses_dp, ep) × expert_tp × cp × fsdp × dp × pp
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  return base * c.expert_tp * c.cp * c.fsdp * c.dp * c.pp
})

const configFormula = computed(() => {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  const parts = []
  
  // Build the base dimension with max() if needed
  const hasAttn = c.attn_tp > 1 || ulysses > 1
  const hasEP = c.ep > 1
  
  if (hasAttn && hasEP) {
    // Show max() formula when both attention and EP are active
    let attnPart = ''
    if (c.attn_tp > 1 && ulysses > 1) {
      attnPart = `ATP${c.attn_tp}×ADP${ulysses}`
    } else if (c.attn_tp > 1) {
      attnPart = `ATP${c.attn_tp}`
    } else if (ulysses > 1) {
      attnPart = `ADP${ulysses}`
    }
    parts.push(`max(${attnPart}, EP${c.ep})`)
  } else if (hasAttn) {
    // Only attention parallelism
    if (c.attn_tp > 1) parts.push(`ATP${c.attn_tp}`)
    if (ulysses > 1) parts.push(`ADP${ulysses}`)
  } else if (hasEP) {
    // Only EP
    parts.push(`EP${c.ep}`)
  }
  
  // Expert TP multiplies (further shards each expert)
  if (c.expert_tp > 1) parts.push(`ETP${c.expert_tp}`)
  if (c.cp > 1) parts.push(`CP${c.cp}`)
  if (c.fsdp > 1) parts.push(`FSDP${c.fsdp}`)
  if (c.dp > 1) parts.push(`DP${c.dp}`)
  if (c.pp > 1) parts.push(`PP${c.pp}`)
  
  if (parts.length === 0) return '1'
  return parts.join(' × ')
})

const memoryEstimate = computed(() => {
  const c = props.config
  const items = []
  
  // Attention Parameters
  let attnParamFactor = 1
  if (c.attn_tp > 1) attnParamFactor *= c.attn_tp
  if (c.fsdp > 1) attnParamFactor *= c.fsdp
  if (c.pp > 1) attnParamFactor *= c.pp
  items.push({
    label: 'Attention Params',
    factor: attnParamFactor > 1 ? `1/${attnParamFactor}` : 'Full'
  })
  
  // Expert/FFN Parameters
  let expertParamFactor = 1
  if (c.expert_tp > 1) expertParamFactor *= c.expert_tp
  if (c.ep > 1) expertParamFactor *= c.ep
  if (c.fsdp > 1) expertParamFactor *= c.fsdp
  if (c.pp > 1) expertParamFactor *= c.pp
  items.push({
    label: 'Expert/FFN Params',
    factor: expertParamFactor > 1 ? `1/${expertParamFactor}` : 'Full'
  })
  
  // Activations
  let actFactor = 1
  if (c.attn_tp > 1) actFactor *= c.attn_tp
  if (c.cp > 1) actFactor *= c.cp
  if (c.pp > 1) actFactor *= c.pp
  items.push({
    label: 'Activations',
    factor: actFactor > 1 ? `1/${actFactor}` : 'Full'
  })
  
  // Optimizer states
  let optFactor = 1
  if (c.fsdp > 1) optFactor *= c.fsdp
  if (c.pp > 1) optFactor *= c.pp
  items.push({
    label: 'Optimizer States',
    factor: optFactor > 1 ? `1/${optFactor}` : 'Full'
  })
  
  return items
})

const activeComms = computed(() => {
  const c = props.config
  const comms = []
  const ulysses = c.ulysses_dp || 1
  
  if (c.attn_tp > 1) {
    comms.push({
      type: 'Attn TP',
      operation: 'AllReduce after attention (QKV, O proj)',
      color: '#ef4444'
    })
  }
  if (ulysses > 1) {
    comms.push({
      type: 'Attention DP',
      operation: 'AllToAll for attention (seq/head redistribution)',
      color: '#dc2626'
    })
  }
  if (c.expert_tp > 1) {
    comms.push({
      type: 'Expert TP',
      operation: 'AllReduce after expert FFN',
      color: '#f97316'
    })
  }
  if (c.dp > 1) {
    comms.push({
      type: 'DP',
      operation: 'AllReduce (gradients)',
      color: '#3b82f6'
    })
  }
  if (c.fsdp > 1) {
    comms.push({
      type: 'FSDP',
      operation: 'AllGather (fwd) + ReduceScatter (bwd)',
      color: '#ec4899'
    })
  }
  if (c.ep > 1) {
    comms.push({
      type: 'EP',
      operation: 'All2All (dispatch & combine)',
      color: '#10b981'
    })
  }
  if (c.cp > 1) {
    comms.push({
      type: 'CP',
      operation: 'AllGather / Ring attention',
      color: '#8b5cf6'
    })
  }
  if (c.pp > 1) {
    comms.push({
      type: 'PP',
      operation: 'P2P Send/Recv (activations)',
      color: '#f59e0b'
    })
  }
  
  return comms
})

const parallelismTypes = [
  {
    key: 'attn_tp',
    abbrev: 'ATP',
    name: 'Attention Tensor Parallelism',
    color: '#ef4444',
    description: 'Splits attention layers (Q, K, V, O projections) across GPUs. Typically uses column-parallel for QKV and row-parallel for O.',
    shards: 'Attention weights (head-wise or column/row)',
    communication: 'AllReduce after attention output projection',
    useCase: 'Large attention heads, typically within a node (NVLink)'
  },
  {
    key: 'ulysses_dp',
    abbrev: 'ADP',
    name: 'Attention DP',
    color: '#dc2626',
    description: 'Uses data parallelism for attention by partitioning sequence across heads. With attn_dp=2 and tp=16, attention effectively becomes tp=8 replicated on 2 groups.',
    shards: 'Sequence dimension across attention heads',
    communication: 'All2All to redistribute seq/head before and after attention',
    useCase: 'Long sequences where attention communication dominates, complements TP'
  },
  {
    key: 'expert_tp',
    abbrev: 'ETP',
    name: 'Expert Tensor Parallelism',
    color: '#f97316',
    description: 'Splits MoE expert FFN layers across GPUs. Can be different from attention TP for optimal efficiency.',
    shards: 'Expert FFN weights (column/row-wise)',
    communication: 'AllReduce after expert FFN',
    useCase: 'Large experts in MoE, can use smaller TP than attention (e.g., DeepSeek uses ETP=1)'
  },
  {
    key: 'dp',
    abbrev: 'DP',
    name: 'Data Parallelism',
    color: '#3b82f6',
    description: 'Replicates the full model on each GPU. Different GPUs process different data batches.',
    shards: 'Data batches',
    communication: 'AllReduce gradients after backward',
    useCase: 'Scaling training throughput when model fits in GPU memory'
  },
  {
    key: 'fsdp',
    abbrev: 'FSDP',
    name: 'Fully Sharded Data Parallel',
    color: '#ec4899',
    description: 'Shards parameters, gradients, and optimizer states across GPUs. Gathers when needed.',
    shards: 'Parameters, gradients, optimizer states',
    communication: 'AllGather before compute, ReduceScatter after',
    useCase: 'Training large models with limited GPU memory'
  },
  {
    key: 'ep',
    abbrev: 'EP',
    name: 'Expert Parallelism',
    color: '#10b981',
    description: 'Distributes MoE experts across GPUs. Tokens are routed to appropriate experts.',
    shards: 'MoE experts (each GPU holds subset of experts)',
    communication: 'All2All for token dispatch/combine',
    useCase: 'Mixture-of-Experts models with many experts'
  },
  {
    key: 'pp',
    abbrev: 'PP',
    name: 'Pipeline Parallelism',
    color: '#f59e0b',
    description: 'Splits model layers into sequential stages. Each GPU handles a portion of the model depth.',
    shards: 'Model layers (stages)',
    communication: 'P2P send/recv between adjacent stages',
    useCase: 'Very deep models, cross-node scaling'
  },
  {
    key: 'cp',
    abbrev: 'CP',
    name: 'Context Parallelism',
    color: '#8b5cf6',
    description: 'Shards the sequence dimension. Useful for very long context lengths.',
    shards: 'Sequence/context tokens',
    communication: 'AllGather KV, Ring attention variants',
    useCase: 'Long sequence training (>32K tokens)'
  }
]

const meshOrder = [
  { key: 'attn_tp', name: 'Attention TP', color: '#ef4444', placement: '① Innermost - within node via NVSwitch for low-latency async TP' },
  { key: 'ulysses_dp', name: 'Attention DP', color: '#dc2626', placement: '① Same level as ATP - DP attention via All2All' },
  { key: 'expert_tp', name: 'Expert TP', color: '#f97316', placement: '① Same level as ATP - tensor parallel for expert FFN' },
  { key: 'ep', name: 'Expert Parallel', color: '#10b981', placement: '② Across rail-optimized leaf switches for best All2All perf' },
  { key: 'cp', name: 'Context Parallel', color: '#8b5cf6', placement: '③ Highest latency priority in CP×FS×DP submesh' },
  { key: 'fsdp', name: 'FSDP', color: '#ec4899', placement: '④ Medium latency priority in CP×FS×DP submesh' },
  { key: 'dp', name: 'Data Parallel', color: '#3b82f6', placement: '⑤ Lowest latency priority in CP×FS×DP submesh' },
  { key: 'pp', name: 'Pipeline Parallel', color: '#f59e0b', placement: '⑥ Outermost - across islands for pipeline stages' }
]
</script>

<style scoped>
.info-panel {
  background: var(--panel-bg);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  margin-top: 24px;
}

.info-panel h2 {
  margin: 0 0 20px 0;
  font-size: var(--font-xl);
  font-weight: 700;
  color: var(--text-primary);
}

.info-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 24px;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 14px;
}

.info-tabs button {
  padding: 10px 18px;
  border: none;
  background: var(--control-bg);
  border-radius: 8px;
  cursor: pointer;
  font-size: var(--font-sm);
  font-weight: 500;
  color: var(--text-secondary);
  transition: all 0.15s;
}

.info-tabs button:hover {
  background: var(--border-color);
  color: var(--text-primary);
}

.info-tabs button:focus-visible {
  box-shadow: var(--focus-ring);
}

.info-tabs button.active {
  background: var(--primary-color);
  color: white;
}

.tab-content {
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Overview Tab */
.overview-grid {
  display: grid;
  gap: 20px;
}

@media (min-width: 768px) {
  .overview-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.overview-card {
  background: var(--control-bg);
  padding: 18px;
  border-radius: 10px;
}

.overview-card h4 {
  margin: 0 0 14px 0;
  font-size: var(--font-sm);
  font-weight: 600;
  color: var(--text-secondary);
}

.config-summary {
  text-align: center;
}

.config-formula {
  font-family: 'SF Mono', Consolas, monospace;
  font-size: var(--font-sm);
  color: var(--text-primary);
  margin-bottom: 10px;
  line-height: var(--leading-relaxed);
}

.config-total {
  font-size: var(--font-xl);
  font-weight: 700;
  color: var(--primary-color);
}

.memory-breakdown {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.memory-item {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-sm);
}

.memory-label {
  color: var(--text-secondary);
}

.memory-factor {
  font-family: 'SF Mono', Consolas, monospace;
  font-weight: 600;
  color: var(--primary-color);
}

.comm-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.comm-item {
  display: flex;
  gap: 10px;
  font-size: var(--font-sm);
}

.comm-type {
  font-weight: 600;
  min-width: 50px;
}

.comm-op {
  color: var(--text-secondary);
}

/* Parallelism Types Tab */
.parallelism-cards {
  display: grid;
  gap: 18px;
}

@media (min-width: 768px) {
  .parallelism-cards {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1200px) {
  .parallelism-cards {
    grid-template-columns: repeat(3, 1fr);
  }
}

.parallelism-card {
  background: var(--control-bg);
  border-radius: 10px;
  padding: 18px;
  border-left: 3px solid transparent;
}

.p-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  border-left: 3px solid;
  padding-left: 12px;
  margin-left: -18px;
}

.p-abbrev {
  padding: 4px 10px;
  border-radius: 4px;
  color: white;
  font-weight: 700;
  font-size: var(--font-xs);
}

.p-name {
  font-weight: 600;
  font-size: var(--font-base);
  color: var(--text-primary);
}

.p-description {
  font-size: var(--font-sm);
  color: var(--text-secondary);
  margin: 0 0 14px 0;
  line-height: var(--leading-relaxed);
}

.p-details {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.p-detail {
  font-size: var(--font-xs);
  color: var(--text-primary);
  line-height: var(--leading-normal);
}

.p-detail strong {
  color: var(--text-secondary);
}

/* Mesh Hierarchy Tab */
.mesh-explanation h4 {
  margin: 0 0 10px 0;
  font-size: var(--font-base);
  font-weight: 600;
  color: var(--text-primary);
}

.mesh-explanation > p {
  margin: 0 0 18px 0;
  font-size: var(--font-sm);
  color: var(--text-secondary);
  line-height: var(--leading-relaxed);
}

.mesh-order {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.mesh-dim {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px;
  background: var(--control-bg);
  border-radius: 10px;
  border-left: 4px solid;
}

.mesh-rank {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--border-color);
  border-radius: 50%;
  font-size: var(--font-xs);
  font-weight: 700;
  color: var(--text-secondary);
}

.mesh-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.mesh-name {
  font-weight: 600;
  font-size: var(--font-base);
}

.mesh-desc {
  font-size: var(--font-xs);
  color: var(--text-secondary);
  line-height: var(--leading-normal);
}

.mesh-note {
  margin-top: 18px;
  padding: 14px;
  background: rgba(79, 70, 229, 0.1);
  border-radius: 10px;
  font-size: var(--font-sm);
  color: var(--text-primary);
  line-height: var(--leading-relaxed);
}

.mesh-note strong {
  color: var(--primary-color);
}
</style>
