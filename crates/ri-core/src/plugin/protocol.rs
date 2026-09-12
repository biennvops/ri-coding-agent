use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const PLUGIN_PROTOCOL_VERSION: &str = "ri.plugin.v1";
pub const MAX_PLUGIN_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("plugin frame exceeds {MAX_PLUGIN_FRAME_BYTES} bytes")]
    FrameTooLarge,
    #[error("invalid plugin JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid JSON-RPC message: {0}")]
    Invalid(&'static str),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    pub params: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcErrorObject>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RpcErrorObject {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum PluginMessage {
    Response(RpcResponse),
    Notification(RpcNotification),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: String,
    pub host: HostIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostIdentity {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub plugin: PluginIdentity,
    pub capabilities: PluginCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginIdentity {
    pub id: String,
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginCapabilities {
    #[serde(default)]
    pub tools: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolsListResult {
    pub tools: Vec<PluginToolDefinition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

pub fn encode_request(id: u64, method: &str, params: Value) -> Result<String, ProtocolError> {
    let line = serde_json::to_string(&RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.into(),
        params,
    })?;
    if line.len() > MAX_PLUGIN_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    Ok(line)
}

pub fn decode_plugin_message(line: &str) -> Result<PluginMessage, ProtocolError> {
    if line.len() > MAX_PLUGIN_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let value: Value = serde_json::from_str(line)?;
    let object = value
        .as_object()
        .ok_or(ProtocolError::Invalid("expected object"))?;
    if object.get("jsonrpc") != Some(&Value::String("2.0".into())) {
        return Err(ProtocolError::Invalid("expected jsonrpc 2.0"));
    }
    if let Some(id) = object.get("id") {
        let id = id
            .as_u64()
            .ok_or(ProtocolError::Invalid("expected unsigned integer id"))?;
        if object.contains_key("method")
            || object.contains_key("result") == object.contains_key("error")
        {
            return Err(ProtocolError::Invalid(
                "response requires exactly one result or error",
            ));
        }
        let error = object
            .get("error")
            .map(|error| serde_json::from_value(error.clone()))
            .transpose()?;
        Ok(PluginMessage::Response(RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: object.get("result").cloned(),
            error,
        }))
    } else {
        Ok(PluginMessage::Notification(serde_json::from_value(value)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_wire_contract_round_trips_and_tolerates_additions() {
        let list = serde_json::json!({"tools":[{"name":"echo","inputSchema":{"type":"object"},"description":"Echo"}]});
        let decoded: ToolsListResult = serde_json::from_value(list.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), list);
        let params = ToolCallParams {
            name: "echo".into(),
            arguments: serde_json::json!({"text":"hello"}),
        };
        assert_eq!(
            serde_json::to_value(params).unwrap(),
            serde_json::json!({"name":"echo","arguments":{"text":"hello"}})
        );
        let result: ToolCallResult =
            serde_json::from_value(serde_json::json!({"content":"hello","future":42})).unwrap();
        assert!(!result.is_error);
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            serde_json::json!({"content":"hello","isError":false})
        );
        let result: ToolCallResult =
            serde_json::from_value(serde_json::json!({"content":"failed","isError":true})).unwrap();
        assert!(result.is_error);
        let list: ToolsListResult = serde_json::from_value(serde_json::json!({"tools":[{"name":"echo","inputSchema":{},"future":true}],"future":true})).unwrap();
        assert!(list.tools[0].description.is_none());
        let _: ToolCallParams =
            serde_json::from_value(serde_json::json!({"name":"echo","arguments":{},"future":true}))
                .unwrap();
    }

    #[test]
    fn wire_round_trips() {
        let params = InitializeParams {
            protocol_version: PLUGIN_PROTOCOL_VERSION.into(),
            host: HostIdentity {
                name: "ri".into(),
                version: "0.1.0".into(),
            },
        };
        let line = encode_request(1, "initialize", serde_json::to_value(params).unwrap()).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&line).unwrap(),
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"ri.plugin.v1","host":{"name":"ri","version":"0.1.0"}}})
        );
        for value in [
            json!({"jsonrpc":"2.0","id":1,"result":null}),
            json!({"jsonrpc":"2.0","id":2,"error":{"code":-1,"message":"bad"}}),
            json!({"jsonrpc":"2.0","method":"event/example","params":{}}),
        ] {
            let message = decode_plugin_message(&value.to_string()).unwrap();
            assert_eq!(serde_json::to_value(message).unwrap(), value);
        }
        let caps: PluginCapabilities =
            serde_json::from_value(json!({"tools":false,"future":42})).unwrap();
        assert_eq!(caps.extra["future"], 42);
    }

    #[test]
    fn rejects_invalid_and_oversized_frames() {
        for value in [
            json!({"jsonrpc":"2.0","id":1,"result":null,"error":{"code":1,"message":"x"}}),
            json!({"jsonrpc":"2.0","id":1}),
            json!({"jsonrpc":"1.0","id":1,"result":0}),
            json!({"jsonrpc":"2.0","id":1,"method":"host/request"}),
            json!([]),
            json!({"jsonrpc":"2.0","id":-1,"result":0}),
            json!({"jsonrpc":"2.0","id":1,"error":null}),
        ] {
            assert!(
                decode_plugin_message(&value.to_string()).is_err(),
                "{value}"
            );
        }
        assert!(encode_request(1, "large", json!("x".repeat(MAX_PLUGIN_FRAME_BYTES))).is_err());
        assert!(decode_plugin_message(&"x".repeat(MAX_PLUGIN_FRAME_BYTES + 1)).is_err());
    }
}
