<template>
  <div class="gpu-details">
    <h3>🔍 GPU {{ gpuId }} Details</h3>
    
    <div class="details-grid">
      <div class="detail-section">
        <h4>Coordinates</h4>
        <div class="coord-list">
          <div 
            v-for="coord in coordinates" 
            :key="coord.key"
            class="coord-item"
          >
            <span class="coord-label" :style="{ color: coord.color }">
              {{ coord.label }}
            </span>
            <span class="coord-value">{{ coord.value }}</span>
          </div>
        </div>
      </div>
      
      <div class="detail-section">
        <h4>What's on this GPU?</h4>
        <div class="content-list">
          <div 
            v-for="item in gpuContent" 
            :key="item.label"
            class="content-item"
          >
            <span class="content-icon">{{ item.icon }}</span>
            <span class="content-text">{{ item.text }}</span>
          </div>
        </div>
      </div>
      
      <div class="detail-section">
        <h4>Communication Groups</h4>
        <div class="group-list">
          <div 
            v-for="group in communicationGroups" 
            :key="group.key"
            class="group-item"
          >
            <span class="group-label" :style="{ color: group.color }">
              {{ group.label }}
            </span>
            <span class="group-value">Group {{ group.groupId }}</span>
            <span class="group-peers">{{ group.peerCount }} peers</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  gpuId: Number,
  config: Object
})

const dimensionColors = {
  attn_tp: '#ef4444',
  ulysses_dp: '#dc2626',
  expert_tp: '#f97316',
  dp: '#3b82f6',
  ep: '#10b981',
  pp: '#f59e0b',
  cp: '#8b5cf6',
  fsdp: '#ec4899'
}

const coords = computed(() => {
  const c = props.config
  // Ulysses DP shares with attention TP: effective_attn = attn_tp * ulysses_dp
  // Base dimension: max(effective_attn, ep) - they share GPUs
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  let remaining = props.gpuId
  
  // Decompose: base (shared attn_tp/ulysses_dp/ep) → expert_tp → cp → fsdp → dp → pp
  const base_coord = remaining % base
  remaining = Math.floor(remaining / base)
  
  // attn_tp and ulysses_dp share the base dimension
  const attn_tp = base_coord % c.attn_tp
  const ulysses_dp = Math.floor(base_coord / c.attn_tp) % ulysses
  const ep = base_coord % c.ep
  
  const expert_tp = remaining % c.expert_tp
  remaining = Math.floor(remaining / c.expert_tp)
  
  const cp = remaining % c.cp
  remaining = Math.floor(remaining / c.cp)
  
  const fsdp = remaining % c.fsdp
  remaining = Math.floor(remaining / c.fsdp)
  
  const dp = remaining % c.dp
  remaining = Math.floor(remaining / c.dp)
  
  const pp = remaining % c.pp
  
  return { base_coord, attn_tp, ulysses_dp, expert_tp, ep, cp, fsdp, dp, pp }
})

const nodeId = computed(() => {
  return Math.floor(props.gpuId / props.config.gpusPerNode)
})

const positionInNode = computed(() => {
  return props.gpuId % props.config.gpusPerNode
})

const coordinates = computed(() => {
  const c = props.config
  const items = []
  const ulysses = c.ulysses_dp || 1
  
  // Always show node info first
  items.push({
    key: 'node',
    label: 'Node',
    value: `${nodeId.value} (GPU ${positionInNode.value} of ${c.gpusPerNode})`,
    color: '#374151'
  })
  
  // Mesh order: TP → Ulysses → EP → CP → FSDP → DP → PP (closest to furthest)
  if (c.attn_tp > 1) {
    items.push({
      key: 'attn_tp',
      label: '① Attention TP',
      value: `${coords.value.attn_tp} / ${c.attn_tp}`,
      color: dimensionColors.attn_tp
    })
  }
  
  if (ulysses > 1) {
    items.push({
      key: 'ulysses_dp',
      label: '① Attention DP',
      value: `${coords.value.ulysses_dp} / ${ulysses}`,
      color: dimensionColors.ulysses_dp
    })
  }
  
  if (c.expert_tp > 1) {
    items.push({
      key: 'expert_tp',
      label: '① Expert TP',
      value: `${coords.value.expert_tp} / ${c.expert_tp}`,
      color: dimensionColors.expert_tp
    })
  }
  
  if (c.ep > 1) {
    items.push({
      key: 'ep',
      label: '② Expert Parallel',
      value: `${coords.value.ep} / ${c.ep}`,
      color: dimensionColors.ep
    })
  }
  
  if (c.cp > 1) {
    items.push({
      key: 'cp',
      label: '③ Context Parallel',
      value: `${coords.value.cp} / ${c.cp}`,
      color: dimensionColors.cp
    })
  }
  
  if (c.fsdp > 1) {
    items.push({
      key: 'fsdp',
      label: '④ FSDP Shard',
      value: `${coords.value.fsdp} / ${c.fsdp}`,
      color: dimensionColors.fsdp
    })
  }
  
  if (c.dp > 1) {
    items.push({
      key: 'dp',
      label: '⑤ Data Parallel',
      value: `${coords.value.dp} / ${c.dp}`,
      color: dimensionColors.dp
    })
  }
  
  if (c.pp > 1) {
    items.push({
      key: 'pp',
      label: '⑥ Pipeline Stage',
      value: `${coords.value.pp} / ${c.pp}`,
      color: dimensionColors.pp
    })
  }
  
  if (items.length === 1) { // Only node info
    items.push({
      key: 'single',
      label: 'Single GPU',
      value: 'No parallelism',
      color: '#666'
    })
  }
  
  return items
})

const gpuContent = computed(() => {
  const c = props.config
  const items = []
  
  // Attention layer parameters
  if (c.attn_tp > 1) {
    items.push({
      icon: '🎯',
      text: `Attention: 1/${c.attn_tp} of Q,K,V,O projections (tensor sharded)`
    })
  } else {
    items.push({
      icon: '🎯',
      text: 'Attention: Full Q,K,V,O projections'
    })
  }
  
  // Expert/FFN parameters
  if (c.ep > 1 && c.expert_tp > 1) {
    items.push({
      icon: '🧩',
      text: `MoE FFN: Expert rank ${coords.value.ep}/${c.ep}, 1/${c.expert_tp} tensor sharded`
    })
  } else if (c.ep > 1) {
    items.push({
      icon: '🧩',
      text: `MoE FFN: Expert rank ${coords.value.ep}/${c.ep} (full expert weights)`
    })
  } else if (c.expert_tp > 1) {
    items.push({
      icon: '🧩',
      text: `FFN: 1/${c.expert_tp} tensor sharded`
    })
  } else {
    items.push({
      icon: '🧩',
      text: 'FFN: Full parameters'
    })
  }
  
  // FSDP sharding
  if (c.fsdp > 1) {
    items.push({
      icon: '📦',
      text: `FSDP: 1/${c.fsdp} of params/grads/optimizer sharded`
    })
  }
  
  // Pipeline stage
  if (c.pp > 1) {
    items.push({
      icon: '🔗',
      text: `Pipeline stage ${coords.value.pp}: layers ${Math.floor(coords.value.pp * 100 / c.pp)}%-${Math.floor((coords.value.pp + 1) * 100 / c.pp)}%`
    })
  }
  
  // Data batch
  if (c.dp > 1 || c.cp > 1) {
    const dpPart = c.dp > 1 ? `1/${c.dp} batch` : 'full batch'
    const cpPart = c.cp > 1 ? `, 1/${c.cp} sequence` : ''
    items.push({
      icon: '📊',
      text: `Processing ${dpPart}${cpPart}`
    })
  } else {
    items.push({
      icon: '📊',
      text: 'Processing full batch'
    })
  }
  
  return items
})

const communicationGroups = computed(() => {
  const c = props.config
  const groups = []
  const ulysses = c.ulysses_dp || 1
  
  if (c.attn_tp > 1) {
    groups.push({
      key: 'attn_tp',
      label: 'Attn TP Group',
      color: dimensionColors.attn_tp,
      groupId: calculateAttnTPGroup(),
      peerCount: c.attn_tp
    })
  }
  
  if (ulysses > 1) {
    groups.push({
      key: 'ulysses_dp',
      label: 'Attention DP Group',
      color: dimensionColors.ulysses_dp,
      groupId: calculateUlyssesDPGroup(),
      peerCount: ulysses
    })
  }
  
  if (c.expert_tp > 1) {
    groups.push({
      key: 'expert_tp',
      label: 'Expert TP Group',
      color: dimensionColors.expert_tp,
      groupId: calculateExpertTPGroup(),
      peerCount: c.expert_tp
    })
  }
  
  if (c.dp > 1) {
    groups.push({
      key: 'dp',
      label: 'DP Group',
      color: dimensionColors.dp,
      groupId: calculateDPGroup(),
      peerCount: c.dp
    })
  }
  
  if (c.fsdp > 1) {
    groups.push({
      key: 'fsdp',
      label: 'FSDP Group',
      color: dimensionColors.fsdp,
      groupId: calculateFSDPGroup(),
      peerCount: c.fsdp
    })
  }
  
  if (c.ep > 1) {
    groups.push({
      key: 'ep',
      label: 'EP Group',
      color: dimensionColors.ep,
      groupId: calculateEPGroup(),
      peerCount: c.ep
    })
  }
  
  if (c.cp > 1) {
    groups.push({
      key: 'cp',
      label: 'CP Group',
      color: dimensionColors.cp,
      groupId: calculateCPGroup(),
      peerCount: c.cp
    })
  }
  
  if (c.pp > 1) {
    groups.push({
      key: 'pp',
      label: 'PP Group',
      color: dimensionColors.pp,
      groupId: calculatePPGroup(),
      peerCount: c.pp
    })
  }
  
  return groups
})

function calculateAttnTPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  // Attn TP group: GPUs with same ulysses_dp, expert_tp, ep, cp, fsdp, dp, pp
  return coords.value.ulysses_dp +
         coords.value.expert_tp * ulysses + 
         coords.value.cp * ulysses * c.expert_tp + 
         coords.value.fsdp * ulysses * c.expert_tp * c.cp +
         coords.value.dp * ulysses * c.expert_tp * c.cp * c.fsdp +
         coords.value.pp * ulysses * c.expert_tp * c.cp * c.fsdp * c.dp
}

function calculateUlyssesDPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  // Ulysses DP group: GPUs with same attn_tp, expert_tp, ep, cp, fsdp, dp, pp
  return coords.value.attn_tp +
         coords.value.expert_tp * c.attn_tp + 
         coords.value.cp * c.attn_tp * c.expert_tp + 
         coords.value.fsdp * c.attn_tp * c.expert_tp * c.cp +
         coords.value.dp * c.attn_tp * c.expert_tp * c.cp * c.fsdp +
         coords.value.pp * c.attn_tp * c.expert_tp * c.cp * c.fsdp * c.dp
}

function calculateExpertTPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  // Expert TP group: GPUs with same base_coord, cp, fsdp, dp, pp
  return coords.value.base_coord + 
         coords.value.cp * base * c.expert_tp + 
         coords.value.fsdp * base * c.expert_tp * c.cp +
         coords.value.dp * base * c.expert_tp * c.cp * c.fsdp +
         coords.value.pp * base * c.expert_tp * c.cp * c.fsdp * c.dp
}

function calculateDPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  // DP group: GPUs with same position except DP dimension
  return coords.value.base_coord + 
         coords.value.expert_tp * base + 
         coords.value.cp * base * c.expert_tp +
         coords.value.fsdp * base * c.expert_tp * c.cp +
         coords.value.pp * base * c.expert_tp * c.cp * c.fsdp
}

function calculateFSDPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  // FSDP group: GPUs with same position except FSDP dimension
  return coords.value.base_coord + 
         coords.value.expert_tp * base + 
         coords.value.cp * base * c.expert_tp +
         coords.value.dp * base * c.expert_tp * c.cp +
         coords.value.pp * base * c.expert_tp * c.cp * c.dp
}

function calculateEPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  // EP group: GPUs that share the same expert set (communicate via All2All for token routing)
  // GPUs with same attn_tp, ulysses_dp, expert_tp, cp, fsdp, dp, pp belong to same EP group
  return coords.value.attn_tp + 
         coords.value.ulysses_dp * c.attn_tp +
         coords.value.expert_tp * effective_attn + 
         coords.value.cp * effective_attn * c.expert_tp + 
         coords.value.fsdp * effective_attn * c.expert_tp * c.cp +
         coords.value.dp * effective_attn * c.expert_tp * c.cp * c.fsdp +
         coords.value.pp * effective_attn * c.expert_tp * c.cp * c.fsdp * c.dp
}

function calculateCPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  // CP group: GPUs with same position except CP dimension
  return coords.value.base_coord + 
         coords.value.expert_tp * base + 
         coords.value.fsdp * base * c.expert_tp +
         coords.value.dp * base * c.expert_tp * c.fsdp +
         coords.value.pp * base * c.expert_tp * c.fsdp * c.dp
}

function calculatePPGroup() {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  // PP group: GPUs with same position except PP dimension
  return coords.value.base_coord + 
         coords.value.expert_tp * base + 
         coords.value.cp * base * c.expert_tp +
         coords.value.fsdp * base * c.expert_tp * c.cp +
         coords.value.dp * base * c.expert_tp * c.cp * c.fsdp
}
</script>

<style scoped>
.gpu-details {
  background: var(--panel-bg);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
}

.gpu-details h3 {
  margin: 0 0 20px 0;
  font-size: var(--font-xl);
  font-weight: 700;
  color: var(--text-primary);
}

.details-grid {
  display: grid;
  gap: 24px;
}

@media (min-width: 768px) {
  .details-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.detail-section h4 {
  margin: 0 0 14px 0;
  font-size: var(--font-sm);
  color: var(--text-secondary);
  font-weight: 600;
}

.coord-list, .content-list, .group-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.coord-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--control-bg);
  border-radius: 8px;
}

.coord-label {
  font-weight: 600;
  font-size: var(--font-sm);
}

.coord-value {
  font-family: 'SF Mono', Consolas, monospace;
  font-size: var(--font-sm);
  color: var(--text-primary);
  font-weight: 500;
}

.content-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  background: var(--control-bg);
  border-radius: 8px;
}

.content-icon {
  font-size: 1.125rem;
  flex-shrink: 0;
}

.content-text {
  font-size: var(--font-sm);
  color: var(--text-primary);
  line-height: var(--leading-relaxed);
}

.group-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: var(--control-bg);
  border-radius: 8px;
  flex-wrap: wrap;
}

.group-label {
  font-weight: 600;
  font-size: var(--font-sm);
  min-width: 100px;
}

.group-value {
  font-family: 'SF Mono', Consolas, monospace;
  font-size: var(--font-sm);
  color: var(--text-primary);
}

.group-peers {
  margin-left: auto;
  font-size: var(--font-xs);
  color: var(--text-secondary);
}
</style>
