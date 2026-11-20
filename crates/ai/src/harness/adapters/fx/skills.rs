//! Port of `harness-fx/src/fx-skills.ts` and libfx's `skills.js`: skills as
//! fx instructions.

use std::collections::HashSet;

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::tools::HarnessSkill;

const MAX_SKILLS: usize = 64;
const MAX_INSTRUCTIONS_BYTES: usize = 64 * 1024;

fn escape_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}

/// `createFxSkillsRuntime({ skills })`: the skills' instructions.
pub(crate) fn create_fx_skills_instructions(skills: &[HarnessSkill]) -> HarnessResult<String> {
    if skills.len() > MAX_SKILLS {
        return Err(HarnessError::new("skills must be an array with at most 64 records").into());
    }
    let mut names = HashSet::new();
    let mut sections = Vec::new();
    for skill in skills {
        if !names.insert(skill.name.as_str()) {
            return Err(HarnessError::new(format!("duplicate skill name: {}", skill.name)).into());
        }
        let resources = skill
            .files
            .iter()
            .flatten()
            .map(|file| {
                format!(
                    "<resource uri=\"{}\">\n{}\n</resource>",
                    escape_attribute(&file.path),
                    file.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let description = if skill.description.is_empty() {
            String::new()
        } else {
            format!("<description>{}</description>", skill.description)
        };
        let section = [
            format!("<skill name=\"{}\">", escape_attribute(&skill.name)),
            description,
            skill.content.clone(),
            resources,
            "</skill>".to_string(),
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
        sections.push(section);
    }
    let instructions = sections.join("\n\n");
    if instructions.len() > MAX_INSTRUCTIONS_BYTES {
        return Err(HarnessError::new(format!(
            "skill instructions exceed the {MAX_INSTRUCTIONS_BYTES} byte libfx limit"
        ))
        .into());
    }
    Ok(instructions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::tools::HarnessSkillFile;

    fn skill(name: &str, description: &str, content: &str, files: Option<Vec<HarnessSkillFile>>) -> HarnessSkill {
        HarnessSkill {
            name: name.into(),
            description: description.into(),
            content: content.into(),
            files,
        }
    }

    #[test]
    fn formats_skills_and_their_resources() {
        let instructions = create_fx_skills_instructions(&[
            skill(
                "review<\"&",
                "Review code",
                "Read the diff.",
                Some(vec![HarnessSkillFile {
                    path: "docs/a\"b.md".into(),
                    content: "Checklist".into(),
                }]),
            ),
            skill("plain", "", "Just do it.", None),
        ])
        .unwrap();
        assert_eq!(
            instructions,
            "<skill name=\"review&lt;&quot;&amp;\">\n<description>Review code</description>\nRead the diff.\n<resource uri=\"docs/a&quot;b.md\">\nChecklist\n</resource>\n</skill>\n\n<skill name=\"plain\">\nJust do it.\n</skill>"
        );
        assert_eq!(create_fx_skills_instructions(&[]).unwrap(), "");
    }

    #[test]
    fn enforces_libfx_limits() {
        let duplicate = create_fx_skills_instructions(&[skill("a", "", "x", None), skill("a", "", "y", None)]);
        assert_eq!(duplicate.unwrap_err().to_string(), "duplicate skill name: a");
        let many: Vec<_> = (0..65).map(|index| skill(&format!("s{index}"), "", "x", None)).collect();
        assert!(create_fx_skills_instructions(&many).is_err());
        let large = create_fx_skills_instructions(&[skill("a", "", &"x".repeat(MAX_INSTRUCTIONS_BYTES), None)]);
        assert!(large.unwrap_err().to_string().contains("byte libfx limit"));
    }
}
