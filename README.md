# Faraya Lodge: Agent-Powered Business Assistant



An AI assistant for **Faraya Lodge**, a fictional family-run mountain guesthouse in Faraya, Lebanon. The assistant, **Nadine**, answers guest questions from the business documents, routes every message to the right handling path, collects booking leads, logs feedback and unanswered questions, and checks its own replies before sending them. It runs on the Google Gemini API and is served through a custom Gradio interface.

📹 **Demo video:** [link](https://drive.google.com/file/d/1eN-qoW26DXGlEhMX87evMosh7SsCBBLa/view?usp=sharing)  




## Features

- **Grounded answers.** Nadine answers only from `business_summary.txt` and `about_business.pdf`, and says so when she doesn't know something instead of guessing.
- **Separate persona and instructions.** Who the agent is (tone, style) and what it must or must not do (guardrails) are written as two separate, clearly labeled blocks.
- **Intent router.** Each message is classified as a *general question*, *lead collection*, or *feedback* before the agent responds, and is handled with its own guidance and set of tools.
- **Tool calling.** Two tools: one saves leads, the other logs feedback and unanswered questions.
- **Short-term memory.** The full conversation history is sent with every message, so details given earlier (dates, number of guests) carry through.
- **Reflection (bonus).** A second model call reviews each draft reply against the instructions and corrects it when needed. Both versions are logged.
- **Custom interface.** A styled chat page with suggested follow-up questions and a toggleable "Agent internals" panel that shows the route, tool calls and reflection result for each message.



## How it works

When a guest sends a message, the **router** first decides what kind of message it is. The message is then passed to the **handler** for that type, which sends it to Gemini together with the system prompt, the conversation history, and the tools that handler is allowed to use. If Gemini decides to call a tool, the tool runs and its result is sent back so the model can finish its answer. Before the reply reaches the guest, the **reflection step** checks the draft and fixes it if something is wrong.

| Intent | Example | Tools available |
|---|---|---|
| `general_question` | "Is breakfast included?" | `record_feedback`, `record_customer_interest` |
| `lead_collection` | "I'd like to book a suite for 2 nights in January" | `record_customer_interest`, `record_feedback` |
| `feedback` | "The heating in our room wasn't working" | `record_feedback` |

The router uses a short Gemini classification call that returns structured JSON. If that call fails, it falls back to keyword matching, so a route is always chosen.

### Tools

Both tools print what they record and append it to a file in `logs/`. Their schemas are defined in MCP style (`name`, `description`, `inputSchema`, `outputSchema`) and converted into Gemini function declarations.

| Tool | Parameters | Called when | Returns |
|---|---|---|---|
| `record_customer_interest` | `email`, `name`, `message` | A guest wants to book or be contacted and has given a name and an email | `status`, `lead_id`, `next_step` (or an error if the email is invalid) |
| `record_feedback` | `question` | The answer isn't in the business documents, or the guest leaves a complaint, suggestion or compliment | `status`, `ticket_id`, `next_step` |



## Project structure

```
business_bot/
├── me/
│   ├── about_business.pdf     # Full business profile
│   └── business_summary.txt   # Short business summary
├── assets/
│   └── nadine.png             # Chat avatar 
├── business_agent.ipynb       # Main notebook: all steps, tests and the demo
├── app.py                     # Same code as the notebook, for deployment
├── requirements.txt
├── .env.example               
└── README.md
```

A `logs/` folder is created at runtime with `leads.jsonl`, `feedback.jsonl` and `reflections.jsonl`.

### Notebook guide

| Section | Content |
|---|---|
| 1–2 | Setup and loading the business documents |
| 3 | **Persona** |
| 4 | **Instructions** |
| 5 | System prompt assembly |
| 6 | Tools and their schemas |
| 7 | Router |
| 8 | Agent loop (history and tool calling) |
| 9 | Route handlers |
| 10 | Reflection step |
| 11 | `respond()`: one full turn from message to reply |
| 12–13 | Router test, scripted demo conversation, and reflection test |
| 14 | Gradio interface |



## Getting started

You need a Gemini API key, which you can get for free from [Google AI Studio](https://aistudio.google.com/apikey).

### Google Colab

1. Upload the `business_bot` folder to `MyDrive/business_bot` in Google Drive.
2. Open `business_agent.ipynb` in Colab.
3. In the 🔑 **Secrets** panel, add a secret named `GEMINI_API_KEY` and turn on **Notebook access**.
4. Run all cells. The first cell installs the libraries and connects Google Drive. The last cell launches the chat and prints a public link.

### Running locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your key to .env
python app.py                 # or open business_agent.ipynb in Jupyter
```

### Configuration

Settings are read from a `.env` file in the project folder. See `.env.example` for the format.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model used for all calls |
| `REFLECTION_ENABLED` | `true` | Turns the reflection step on or off |



## Example conversation

> **Guest:** We're a family of 4, we'd like to come for 2 nights in late December.

> **Nadine:** We'd love to host your family! Our Family Room sleeps up to four guests for $120 per night, including breakfast. Could you share your name and email, and your exact dates if you have them? Rami will check availability and email you within 24 hours.


> **Guest:** I'm Sara Khalil, sara.khalil@gmail.com


> **Nadine:** Thank you, Sara! I've passed your details on for a 2-night stay for 4 people in late December. Rami will email you within 24 hours to confirm availability and the deposit.

Behind the scenes, the second message was routed to `lead_collection` and triggered:

```
record_customer_interest(name="Sara Khalil", email="sara.khalil@gmail.com",
                         message="Family of 4, 2-night stay in late December (Family Room).")
```




## Notes

- The assignment mentions `google.generativeai`. This project uses `google-genai`, Google's current Python SDK for the same Gemini API, which replaces the older library.
- Faraya Lodge, its team and all of its details are fictional.
