use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl ThinkingLevel {
    pub const ALL: [Self; 6] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

impl fmt::Display for ThinkingLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("invalid thinking level {value:?}; expected off, minimal, low, medium, high, or xhigh")]
pub struct ThinkingLevelError {
    pub value: String,
}

impl FromStr for ThinkingLevel {
    type Err = ThinkingLevelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            _ => Err(ThinkingLevelError {
                value: value.to_owned(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_and_display_canonically() {
        for level in ThinkingLevel::ALL {
            assert_eq!(level.as_str().parse::<ThinkingLevel>().unwrap(), level);
            assert_eq!(level.to_string(), level.as_str());
        }
        assert_eq!(
            "HIGH".parse::<ThinkingLevel>().unwrap(),
            ThinkingLevel::High
        );
    }

    #[test]
    fn invalid_levels_are_actionable() {
        let error = "extreme".parse::<ThinkingLevel>().unwrap_err();
        assert_eq!(error.to_string(), "invalid thinking level \"extreme\"; expected off, minimal, low, medium, high, or xhigh");
    }

    #[test]
    fn levels_are_ordered_from_off_to_xhigh() {
        assert!(ThinkingLevel::Off < ThinkingLevel::Minimal);
        assert!(ThinkingLevel::Minimal < ThinkingLevel::Low);
        assert!(ThinkingLevel::Low < ThinkingLevel::Medium);
        assert!(ThinkingLevel::Medium < ThinkingLevel::High);
        assert!(ThinkingLevel::High < ThinkingLevel::XHigh);
    }
}
