//! Host-side capability composition, not external plugin loading.

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

pub mod protocol;
