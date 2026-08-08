use super::ContextBundle;

pub const BUILT_IN_SYSTEM_PROMPT: &str = "You are ri, a coding agent.\n\n\
You can inspect and modify files and run commands using the provided tools.\n\
Use tools when necessary to complete the user's task.\n\
Keep changes focused and verify relevant work when practical.\n";

pub fn build_system_prompt(context: &ContextBundle) -> String {
    let mut prompt = String::from(BUILT_IN_SYSTEM_PROMPT);
    prompt.push('\n');
    prompt.push_str("Tool workspace: ");
    prompt.push_str(&context.launch_cwd.display().to_string());
    prompt.push('\n');
    prompt.push_str("Project root: ");
    prompt.push_str(&context.project_root.display().to_string());
    prompt.push('\n');

    if !context.files.is_empty() {
        prompt.push('\n');
        prompt.push_str("Project instructions follow.\n");
        for file in &context.files {
            prompt.push('\n');
            prompt.push_str("<agents-context path=\"");
            prompt.push_str(&escape_xml_attribute(&file.path.display().to_string()));
            prompt.push_str("\" bytes=\"");
            prompt.push_str(&file.content.len().to_string());
            prompt.push_str("\">\n");
            prompt.push_str(&file.content);
            if !file.content.ends_with('\n') {
                prompt.push('\n');
            }
            prompt.push_str("</agents-context>\n");
        }
    }

    prompt
}

fn escape_xml_attribute(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::context::load_context_with_home;

    #[test]
    fn builds_one_deterministic_prompt_with_metadata_and_ordered_sections() {
        let home = unique_test_dir("prompt-home");
        let root = unique_test_dir("prompt-root");
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir_all(home.join(".ri/agent")).unwrap();
        fs::write(home.join(".ri/agent/AGENTS.md"), "global instruction").unwrap();
        fs::write(root.join("AGENTS.md"), "root instruction").unwrap();
        fs::write(nested.join("AGENTS.md"), "nested instruction\n").unwrap();

        let context = load_context_with_home(&nested, &root, Some(&home)).unwrap();
        let prompt = build_system_prompt(&context);

        assert_eq!(prompt.matches("You are ri, a coding agent.").count(), 1);
        assert_eq!(prompt.matches("Tool workspace:").count(), 1);
        assert_eq!(prompt.matches("Project root:").count(), 1);
        assert_eq!(prompt.matches("<agents-context").count(), 3);
        assert!(
            prompt.find("global instruction").unwrap() < prompt.find("root instruction").unwrap()
        );
        assert!(
            prompt.find("root instruction").unwrap() < prompt.find("nested instruction").unwrap()
        );
        for file in &context.files {
            assert!(prompt.contains(&format!("path=\"{}\"", file.path.display())));
            assert!(prompt.contains(&file.content));
        }
        assert!(prompt.contains("bytes=\"16\""));
        assert!(prompt.contains("</agents-context>"));

        remove_test_dir(home);
        remove_test_dir(root);
    }

    #[test]
    fn escapes_paths_without_mutating_instruction_content() {
        let root = unique_test_dir("prompt-&\"<>");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("AGENTS.md"), "raw *markdown* & <tag>\n").unwrap();

        let context = load_context_with_home(&root, &root, None).unwrap();
        let prompt = build_system_prompt(&context);

        assert!(prompt.contains("&amp;") || prompt.contains("&quot;") || prompt.contains("&lt;"));
        assert!(prompt.contains("raw *markdown* & <tag>\n"));
        let unescaped_path = "path=\"".to_owned() + &root.display().to_string();
        assert!(!prompt.contains(&unescaped_path));

        remove_test_dir(root);
    }

    #[test]
    fn prompt_without_files_still_contains_the_builtin_context() {
        let root = unique_test_dir("prompt-empty");
        fs::create_dir_all(&root).unwrap();
        let context = load_context_with_home(&root, &root, None).unwrap();

        let prompt = build_system_prompt(&context);

        assert!(prompt.starts_with(BUILT_IN_SYSTEM_PROMPT));
        assert!(prompt.contains(&format!("Tool workspace: {}", context.launch_cwd.display())));
        assert!(!prompt.contains("agents-context"));
        remove_test_dir(root);
    }

    #[test]
    fn disabled_context_keeps_builtin_prompt_and_metadata() {
        let root = unique_test_dir("prompt-disabled");
        fs::create_dir_all(&root).unwrap();
        let context = ContextBundle::disabled(root.clone(), root.clone());

        let prompt = build_system_prompt(&context);

        assert!(prompt.starts_with(BUILT_IN_SYSTEM_PROMPT));
        assert!(!prompt.contains("agents-context"));
        assert!(prompt.contains(&root.display().to_string()));
        remove_test_dir(root);
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ri-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn remove_test_dir(path: PathBuf) {
        fs::remove_dir_all(path).unwrap();
    }
}
