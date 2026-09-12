//! Built-in capability composition and explicitly invoked external-plugin lifecycle.

use std::sync::Arc;

use crate::tools::{builtin_tool_registry, ToolRegistry};

#[derive(Clone, Debug, Default)]
pub struct PluginRegistry {
    tools: Arc<ToolRegistry>,
}

impl PluginRegistry {
    pub fn new(tools: Arc<ToolRegistry>) -> Self {
        Self { tools }
    }

    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }
}

pub fn builtin_plugins() -> PluginRegistry {
    PluginRegistry::new(Arc::new(builtin_tool_registry()))
}

pub mod manifest;
mod process;
pub mod protocol;

pub use manifest::{
    load_plugin_manifest, LoadedPluginManifest, PluginEntrypoint, PluginManifest,
    PluginManifestError, PLUGIN_MANIFEST_VERSION,
};
pub use process::{
    PluginDiagnostics, PluginProcess, PluginProcessError, MAX_PLUGIN_STDERR_BYTES,
    PLUGIN_SHUTDOWN_TIMEOUT, PLUGIN_STARTUP_TIMEOUT,
};
pub use protocol::{
    PluginCapabilities, PluginIdentity, MAX_PLUGIN_FRAME_BYTES, PLUGIN_PROTOCOL_VERSION,
};

#[cfg(test)]
mod tests {
    #[test]
    fn normal_bootstrap_remains_builtin_only() {
        assert_eq!(
            super::builtin_plugins().tools().names(),
            ["read", "write", "edit", "bash"]
        );
        assert!(super::PluginRegistry::default().tools().names().is_empty());
    }
}

mod tools;

#[cfg(test)]
pub(crate) use process::tests::Fixture;
pub use protocol::{PluginToolDefinition, ToolCallParams, ToolCallResult, ToolsListResult};
pub use tools::{
    ExternalToolError, ExternalToolSet, MAX_EXTERNAL_TOOLS_PER_PLUGIN,
    MAX_EXTERNAL_TOOL_DESCRIPTION_BYTES, MAX_EXTERNAL_TOOL_NAME_BYTES,
};
