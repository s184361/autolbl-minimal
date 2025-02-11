import textgrad as tg
from textgrad.tasks import load_task
import os
import random
import time

"""### Utilities to run the code, and test cases"""

# We'll use below utilities to run a python function.
from IPython.core.interactiveshell import InteractiveShell

def run_function_in_interpreter(func_code):
    #raise Exception("This function will run the code returned by GPT-4o. Remove this if you'd like to run the code!")
    interpreter = InteractiveShell.instance()

    interpreter.run_cell(func_code, store_history=False, silent=True)

    func_name = func_code.split("def ")[1].split("(")[0].strip()
    func = interpreter.user_ns[func_name]

    return func


def test_longest_increasing_subsequence(fn):
    nums = [10, 22, 9, 33, 21, 50, 41, 60]
    assert fn(nums) == 5

    nums = [7, 2, 1, 3, 8, 4, 9, 6, 5]
    assert fn(nums) == 4

    nums = [5, 4, 3, 2, 1]
    assert fn(nums) == 1

    nums = [1, 2, 3, 4, 5]
    assert fn(nums) == 5

    nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    assert fn(nums) == 4

    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    assert fn(nums) == 4

    nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert fn(nums) == 6

    nums = [7, 7, 7, 7, 7, 7, 7]
    assert fn(nums) == 1

    nums = [20, 25, 47, 35, 56, 68, 98, 101, 212, 301, 415, 500]
    assert fn(nums) == 11

    nums = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert fn(nums) == 1

    print("All test cases passed!")


problem_text = """Input prompt for VLM classifier:

Problem Statement:
Given the description of the the object you look for you need to find them in pictures, draw bouding boxes around them and label them. The objective is to maximize the accuracy of the model in identifying the objects.

Input:
Prompt for the VLM classifier. Description of the object to look for in the images. It can be a single word or a phrase. Sometimes shorter prompts work better sometimes longer prompts work better. The prompt should be designed to maximize the accuracy of the model in identifying the objects.

Output:
The output should be the accuracy of the model in identifying the objects. The accuracy should be between 0 and 1. The higher the accuracy the better the model is at identifying the objects."""

initial_solution = """
defect anomaly scratch crack split knot dead knot in wood
"""

# Test the function with a random test case
size = 10000  # Adjust the size as needed
min_value = 1
max_value = 1000

os.environ["ANTHROPIC_API_KEY"] = (
    "os.getenv("ANTHROPIC_API_KEY", "")"
)
llm_engine = tg.get_engine("haiku")
tg.set_backward_engine("haiku")

# Code is the variable of interest we want to optimize -- so requires_grad=True
code = tg.Variable(value=initial_solution,
                   requires_grad=True,
                   role_description="code instance to optimize")

# We are not interested in optimizing the problem -- so requires_grad=False
problem = tg.Variable(problem_text,
                      requires_grad=False,
                      role_description="the coding problem")

# Let TGD know to update code!
optimizer = tg.TGD(parameters=[code])

# The system prompt that will guide the behavior of the loss function.
loss_system_prompt = "You are a smart language model that evaluates input prompts for a vision language model that looks for the defects in wood samples. You do not solve problems or propose new prompts, only evaluate existing prompt critically and give very concise feedback."
loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")

# The instruction that will be the prefix
instruction = """Think about the problem and the prompt to the VLM. Does the code solve the problem? What is the accuracy of VLM classification?"""

# The format string and setting up the call
format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
format_string = format_string.format(instruction=instruction)

fields = {"problem": None, "code": None}
formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                  format_string=format_string,
                                                  fields=fields,
                                                  system_prompt=loss_system_prompt)

# Finally, the loss function
def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
    inputs = {"problem": problem, "code": code}

    return formatted_llm_call(inputs=inputs,
                              response_role_description=f"evaluation of the {code.get_role_description()}")

# Let's do the forward pass for the loss function.
loss = loss_fn(problem, code)
print(loss.value)

# Let's visualize our computation graph.
loss.generate_graph()

# Let's look at the gradients!
loss.backward()
print(code.gradients)

# Let's update the code
optimizer.step()

# Hopefully, we should get much better runtime!
longest_increasing_subsequence = run_function_in_interpreter(code.value)

start_time = time.time()
lis = longest_increasing_subsequence(nums)
end_time = time.time()

print(f"Longest Increasing Subsequence Length: {lis}")
print(f"Runtime: {end_time - start_time:.5f} seconds")

test_longest_increasing_subsequence(longest_increasing_subsequence)

# Let's do one more iteration
optimizer.zero_grad()
loss = loss_fn(problem, code)
loss.backward()
optimizer.step()

longest_increasing_subsequence = run_function_in_interpreter(code.value)

start_time = time.time()
lis = longest_increasing_subsequence(nums)
end_time = time.time()

print(f"Longest Increasing Subsequence Length: {lis}")
print(f"Runtime: {end_time - start_time:.5f} seconds")

test_longest_increasing_subsequence(longest_increasing_subsequence)

"""## Optimized code, much faster!"""

print(code.value)

