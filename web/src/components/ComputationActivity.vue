<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  label: string;
  detail: string;
  elapsedMs: number;
  progress: number | null;
}>();

const elapsedLabel = computed(() => `${(props.elapsedMs / 1000).toFixed(1)}s`);
const progressWidth = computed(() => `${Math.max(0, Math.min(1, props.progress ?? 0)) * 100}%`);
</script>

<template>
  <div class="computation-activity" role="status" aria-live="polite">
    <div class="computation-orbit" aria-hidden="true">
      <i></i><i></i><i></i><b></b>
    </div>
    <div class="computation-copy">
      <div><span>{{ label }}</span><strong>{{ elapsedLabel }}</strong></div>
      <p>{{ detail }}</p>
      <div class="computation-progress" :class="{ 'is-indeterminate': progress === null }" aria-hidden="true">
        <i :style="progress === null ? undefined : { width: progressWidth }"></i>
      </div>
    </div>
  </div>
</template>
