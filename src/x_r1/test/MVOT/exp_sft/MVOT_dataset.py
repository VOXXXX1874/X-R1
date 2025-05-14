# Import libraries to read json files and manipulate data
import json
# Import OpenAI API
from openai import AsyncOpenAI
import asyncio

# Read the json file train.json
with open("test.json", "r") as f:
    math_qa_dataset = json.load(f)


prompt_MVOT_generation = (
    "We try to improve LLM's performance on solving math problems."
    "As we know, majority voting can enhance the performance of LLMs and chain of thought is a common way to solve math problems."
    "However, we find that the potential of majority voting on the intermediate steps of chain of thought is not fully explored."
    "In this task, we will provide you with a math problem and its solution."
    "Your task is to formulate the solution into a desired format."
    "In this format, there is several steps. In each step, there is a reasoning process and an answer."
    "The answer along should be enough for the next step and the reasoning process should be removable."
    "Here is an example:\n"
    "### Problem:\n"
    "What are all values of $p$ such that for every $q>0$, we have $$\\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ Express your answer in interval notation in decimal form.\n"
    "### Solution:\n"
    "First we'll simplify that complicated expression. We attempt to factor the numerator of the left side: \\begin{align*} pq^2+p^2q+3q^2+3pq &= q(pq + p^2 + 3q + 3p) \\\\ &= q[ p(q+p) + 3(q+p) ] \\\\ &= q(p+3)(q+p). \\end{align*}Substituting this in for the numerator in our inequality gives $$\\frac{3q(p+3)(p+q)}{p+q}>2p^2q.$$We note that left hand side has $p+q$ in both the numerator and denominator. We can only cancel these terms if $p+q \\neq 0.$ Since we're looking for values of $p$ such that the inequality is true for all $q > 0,$ we need $p \\geq 0$ so that $p + q \\neq 0.$ Also because this must be true for every $q>0$, we can cancel the $q$'s on both sides. This gives \\begin{align*} 3(p+3)&>2p^2\\Rightarrow\\\\ 3p+9&>2p^2 \\Rightarrow\\\\ 0&>2p^2-3p-9. \\end{align*}Now we must solve this quadratic inequality. We can factor the quadratic as $2p^2-3p-9=(2p+3)(p-3)$. The roots are $p=3$ and $p=-1.5$. Since a graph of this parabola would open upwards, we know that the value of $2p^2 - 3p - 9$ is negative between the roots, so the solution to our inequality is $-1.5<p<3.$ But we still need $0 \\leq p,$ so in interval notation the answer is $\\boxed{[0,3)}$."
    "### Formatted Solution:\n"
    "Step 1:\n"
    "<think> First we'll simplify that complicated expression. We attempt to factor the numerator of the left side: \\begin{align*} pq^2+p^2q+3q^2+3pq &= q(pq + p^2 + 3q + 3p) \\\\ &= q[ p(q+p) + 3(q+p) ] \\\\ &= q(p+3)(q+p). \\end{align*} </think>\n"
    "<answer> Numerator of the left side: $pq^2+p^2q+3q^2+3pq = q(p+3)(q+p)$ </answer>\n"
    "Step 2:\n"
    "<think> Substituting this in for the numerator in our inequality gives $$\\frac{3q(p+3)(p+q)}{p+q}>2p^2q.$$We note that left hand side has $p+q$ in both the numerator and denominator. We can only cancel these terms if $p+q \\neq 0.$ Since we're looking for values of $p$ such that the inequality is true for all $q > 0,$ we need $p \\geq 0$ so that $p + q \\neq 0.$\n"
    "<answer> We can derive $p \\geq 0$ </answer>\n"
    "Step 3:\n"
    "<think> Also because this must be true for every $q>0$, we can cancel the $q$'s on both sides. This gives \\begin{align*} 3(p+3)&>2p^2\\Rightarrow\\\\ 3p+9&>2p^2 \\Rightarrow\\\\ 0&>2p^2-3p-9. \\end{align*} </think>\n"
    "<answer> The inequality can be simplified to $2p^2-3p-9<0$ </answer>\n"
    "Step 4:\n"
    "<think> Now we must solve this quadratic inequality. We can factor the quadratic as $2p^2-3p-9=(2p+3)(p-3)$. The roots are $p=3$ and $p=-1.5$. Since a graph of this parabola would open upwards, we know that the value of $2p^2 - 3p - 9$ is negative between the roots, so the solution to our inequality is $-1.5<p<3.$ But we still need $0 \\leq p,$ so in interval notation the answer is $\\boxed{[0,3)}$.</think>\n"
    "<final> $\\boxed{[0,3)}$ </final>\n"
    "\n"
    "You can format the solution in arbitrary steps depending on the complexity. The math problem you are given is:\n"
    "### Problem:\n"
    )


async def generate_questions(qa, prompt, result_list, exception_list, sem, client):
    async with sem:
        print(f"Generate formatted solution")
        tmp_user_prompt = prompt + qa["problem"] + "\n### Solution:\n" + qa["solution"]
        try:
            # generate response
            response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are a math expert."
                    },
                    {
                        "role": "user",
                        "content": tmp_user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": "### Formatted Solution:\n"
                    }
                ],
                stream = False,
                temperature = 0.5,
            )
            # Read the response
            output = response.choices[0].message.content
            qa['messages'] = [
                {
                    "role": "user",
                    "content": qa["problem"]
                },
                {
                    "role": "assistant",
                    "content": output
                }
            ]
            # Append the result to the result list
            result_list.append(qa)
        except Exception as e:
            print(f"Error in generation")
            print(e)
            exception_list.append(qa)

    print(f"Finish generation")


# A function to generate the dataset
async def generate_dataset(qa_dataset, prompt):
    # Initialize deepseek client
    client = AsyncOpenAI(api_key="sk-6a81291d661d46f9b4cd18f8cc43b693", base_url="https://api.deepseek.com")
    # Control the number of tasks running concurrently
    sem = asyncio.Semaphore(100)
    # Create a list to store the results
    result_list = []
    # Create a exception list to store the exception
    exception_list = []


    print("Start generating questions")
    # Create a list of tasks to generate questions for each qa
    tasks = [generate_questions(qa, prompt, result_list, exception_list, sem, client) for qa in qa_dataset]

    # Run the tasks
    await asyncio.gather(*tasks)

    # Save it as result.json
    with open("result.json", "w") as f:
        json.dump(result_list, f)

    # Save the exception into a json file
    with open("exception.json", "w") as f:
        json.dump(exception_list, f)

# Run the main function
asyncio.run(generate_dataset(math_qa_dataset, prompt_MVOT_generation))