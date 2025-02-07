import textgrad as tg
from textgrad.tasks import load_task
import os

os.environ["ANTHROPIC_API_KEY"] = (
    "os.getenv("ANTHROPIC_API_KEY", "")"
)
llm_engine = tg.get_engine("haiku")
tg.set_backward_engine("haiku")

_, val_set, _, eval_fn = load_task("BBH_object_counting", llm_engine)
question_str, answer_str = val_set[0]
question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)
system_prompt = tg.Variable("You are a concise LLM. Think step by step.",
                            requires_grad=True,
                            role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))

prediction = model(question)
loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))

loss.backward()
optimizer.step()
prediction = model(question)