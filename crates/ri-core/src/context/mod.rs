pub mod accounting;
pub mod agents;
pub mod project;
pub mod prompt;

pub use accounting::{
    automatic_trigger, compaction_target, input_budget, ConservativeTokenEstimator, ContextUsage,
    GenericTokenEstimator, TokenEstimator, UsageSource, AUTO_COMPACTION_TARGET_PERCENT,
    AUTO_COMPACTION_TRIGGER_PERCENT, COMPACTION_MAX_OUTPUT_TOKENS, DEFAULT_RESERVED_OUTPUT_TOKENS,
};
pub use agents::{
    load_context, load_context_with_home, ContextBundle, ContextError, ContextFileKind,
    LoadedContextFile, MAX_CONTEXT_FILE_BYTES, MAX_TOTAL_CONTEXT_BYTES,
};
pub use project::{
    canonicalize_launch_cwd, discover_project, discover_project_root, ProjectError, ProjectLayout,
};
pub use prompt::{build_system_prompt, BUILT_IN_SYSTEM_PROMPT};
