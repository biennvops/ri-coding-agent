use std::sync::Arc;

use super::{bash::BashTool, edit::EditTool, read::ReadTool, write::WriteTool, Tool, ToolRegistry};

pub fn builtin_tool_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    let tools: [Arc<dyn Tool>; 4] = [
        Arc::new(ReadTool),
        Arc::new(WriteTool),
        Arc::new(EditTool),
        Arc::new(BashTool),
    ];
    for tool in tools {
        registry.register(tool).expect("unique built-in tool names");
    }
    registry
}
