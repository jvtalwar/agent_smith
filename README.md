# 🕶️ Agent-Smith: The Fastino Gmail Reply Agent

**Agent-Smith** is an experimental, LLM-powered Gmail reply agent that automates and personalizes your email workflow.  It identifies which emails need responses, generates draft replies with transparent bullet-point reasoning, and learns from your feedback to refine tone, style, and decision-making over time.  

---

## 🚀 Getting Started

The **Agent-Smith Gmail Agent** is currently in **testing mode**, and access is limited to approved users.

### 1. Request Access
To participate in testing:
1. Send me an email at **james.talwar@gmail.com** expressing your interest.  
2. Include the **Gmail account** you’d like to use for testing.  
3. Once approved, I’ll add your Gmail address as a test user and send you a `credentials.json` file.  

> ⚠️ **Important:** Access is limited during this phase for safety and controlled evaluation of Gmail API integrations.

---

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/agent-smith.git
cd agent-smith
```

### 3. Set Up the Environment (using [`uv`](https://docs.astral.sh/uv/getting-started/features/#projects))
Install dependencies and initialize the package:
```bash
uv pip install -e .
```

### 4. Add Credentials
After receiving your `credentials.json`:
1. Move the file into the **project root directory**.  
2. On first run, you’ll be asked to **authenticate Gmail access** — please accept this prompt.

### 5. Create a `.env` File
Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## ⚙️ Configuration

You can customize **Agent-Smith** by editing the `agent_config.yaml` file.

| Setting | Description | Example |
|----------|--------------|----------|
| `max_threads` | Number of most recent Gmail threads to process (excluding spam/trash). | `25` |
| `backend_provider` | Backend for model inference (currently only `openai` - `anthropic` up next). | `openai` |
| `model_name` | OpenAI model for text generation. | `gpt-4.1-nano-2025-04-14`, `gpt-5-nano-2025-08-07`, etc.|
| `num_judges` | Number of LLMs to use when deciding if an email requires a reply (max 5). | `5` |
| `embedding_model` | Embedding model for style and context retrieval. | `text-embedding-3-small` |

> 💡 **Tip:**  
> `gpt-4.1-nano` is **faster and cheaper**, while `gpt-5-nano` provides **better reasoning quality** at higher token cost (and is slower).

---

## 🧠 Running Agent-Smith

Once configured, you can control the agent through simple CLI commands.

### Run the Agent
```bash
agent-smith run
```
- Scans your Gmail inbox for recent threads  
- Identifies which emails warrant replies  
- Generates draft responses and creates Gmail drafts for your review  

### Consolidate Offline Memory
```bash
agent-smith consolidate
```
- Updates internal style and tone memories based on feedback  
- Prunes expired or low-weight memories for better performance and efficiency  

### View Agent Memory
```bash
agent-smith show-memories
```
- Displays all stored **style**, **contact**, **thread**, and **outcome** memories  
- Useful for inspecting how the agent is adapting to your writing style  

---

## 🧩 Command Reference

| Command | Description |
|----------|-------------|
| `run` | Run the Gmail reply agent: identifies emails needing replies, generates drafts, and saves them to Gmail. |
| `show-memories` | Display all stored memories (style, contact, thread, outcome). |
| `consolidate` | Perform offline consolidation of memories to align the agent with your communication style. |

---

## 🔒 Privacy & Access Notes

- Uses **Gmail API (OAuth)** for message reading and draft creation — it **never sends emails automatically**.  
- All API keys, credentials, and memory files remain **local** to your environment.  
- During testing, only **approved Gmail accounts** will function with the system.  

---

## 🧭 Roadmap

- [ ] Expand backend providers (Anthropic, Gemini, etc.)  
- [ ] Infer user feedback directly from gmail drafts (as opposed to per draft now)
- [ ] Enable continuous background mode  

---

## 📧 Contact

For questions, testing access, or collaboration inquiries:  
**James Talwar** — james.talwar@gmail.com

