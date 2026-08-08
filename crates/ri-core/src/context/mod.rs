pub mod agents;
pub mod project;
pub mod prompt;

pub use agents::{
    load_context, load_context_with_home, ContextBundle, ContextError, ContextFileKind,
    LoadedContextFile, MAX_CONTEXT_FILE_BYTES, MAX_TOTAL_CONTEXT_BYTES,
};
pub use project::{
    canonicalize_launch_cwd, discover_project, discover_project_root, ProjectError, ProjectLayout,
};
pub use prompt::{build_system_prompt, BUILT_IN_SYSTEM_PROMPT};
