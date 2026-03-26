# Claude Code System Prompt (Reconstructed)

Extracted from the Claude Code binary (v2.1.84, 194MB Mach-O ARM64) and visible system-reminder messages in the conversation.

The system prompt is built dynamically from template strings in the minified JavaScript. Key sections:

---

## Identity

```
You are Claude Code, Anthropic's official CLI for Claude.
You are an interactive agent that helps users with software engineering tasks.
Use the instructions below and the tools available to you to assist the user.
```

## Security Policy

```
IMPORTANT: Assist with authorized security testing, defensive security, CTF
challenges, and educational contexts. Refuse requests for destructive techniques,
DoS attacks, mass targeting, supply chain compromise, or detection evasion for
malicious purposes. Dual-use security tools (C2 frameworks, credential testing,
exploit development) require clear authorization context: pentesting engagements,
CTF competitions, security research, or defensive use cases.
```

```
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are
confident that the URLs are for helping the user with programming.
```

## Sections (from binary analysis)

The prompt is composed of these major sections:

1. **# Doing tasks** — Guidelines for software engineering tasks
2. **# Executing actions with care** — Risk assessment for reversible vs irreversible actions
3. **# Using your tools** — When to use each built-in tool (Read, Write, Edit, Bash, Glob, Grep, etc.)
4. **# Tone and style** — Concise, no emojis, markdown formatting
5. **# Output efficiency** — Be direct, skip filler words
6. **# Environment** — Current working directory, platform, shell, OS version, model info

## Built-in Tools

From the binary, the registered tools are:

| Tool | Purpose |
|------|---------|
| `Read` | Read files |
| `Write` | Write new files |
| `Edit` | Edit existing files (string replacement) |
| `Bash` | Execute shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `Agent` | Spawn sub-agents |
| `Skill` | Invoke slash commands |
| `WebFetch` | Fetch web content |
| `WebSearch` | Search the web |
| `NotebookEdit` | Edit Jupyter notebooks |
| `AskUserQuestion` | Ask the user a question |
| `SendMessage` | Send message to another agent |
| `TaskCreate` / `TaskUpdate` / `TaskGet` / `TaskList` | Task management |
| `ToolSearch` | Search for deferred tools |

## Hook Events

The binary defines these hook events:

```
PreToolUse, PostToolUse, PostToolUseFailure,
Notification, UserPromptSubmit, SessionStart, SessionEnd,
Stop, StopFailure, SubagentStart, SubagentStop,
PreCompact, PostCompact, PermissionRequest,
Setup, TeammateIdle, TaskCreated, TaskCompleted,
Elicitation, ElicitationResult, ConfigChange,
WorktreeCreate, WorktreeRemove, InstructionsLoaded,
CwdChanged, FileChanged
```

## Agent Types

```
general-purpose, statusline-setup, Explore, Plan, claude-code-guide
```

## Model References

```
claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001
```

Also references: `anthropic.ModelClaudeOpus4_6`, `anthropic.ModelClaudeSonnet4_6` (Go SDK)

## Skill Discovery

Skills are loaded from:
- `~/.claude/skills/<name>/SKILL.md` (personal)
- `.claude/skills/<name>/SKILL.md` (project)
- Plugin skills (namespaced: `plugin:skill`)

## Key Behavioral Rules (from binary strings)

```
- Do NOT use the Bash to run commands when a relevant dedicated tool is provided
- Only create commits when requested by the user
- NEVER update the git config
- NEVER run destructive git commands unless explicitly requested
- NEVER skip hooks (--no-verify) unless explicitly asked
- Do not create files unless absolutely necessary
- Avoid giving time estimates
- Be careful not to introduce security vulnerabilities
- Don't add features beyond what was asked
```

## Dynamic Sections

The prompt includes dynamic content:
- Current working directory and git status
- Available skills list (from SKILL.md files)
- CLAUDE.md content (project instructions)
- Available MCP servers and their tools
- System date
- Model capabilities
- Permission mode

## Binary Details

```
Path: ~/.local/share/claude/versions/2.1.84
Size: 194 MB
Type: Mach-O 64-bit executable arm64
Runtime: Bun (JavaScript bundled as Single Executable Application)
Contains: Minified JavaScript with embedded documentation,
          SDK examples (Python, TypeScript, Go), and tool definitions
```
