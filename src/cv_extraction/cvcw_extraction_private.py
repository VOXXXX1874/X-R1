# Import libraries to read json files and manipulate data
import json
# Import OpenAI API
from openai import AsyncOpenAI
import asyncio

# Read the json file train.json
with open("MATH-500/raw/test.json", "r") as f:
    math_qa_dataset = json.load(f)


prompt_cv_extraction_generation = """We are using Reinforcement Learning to make LLM learn to solve math problems. We find that reward on problem solving process is more efficient than reward on final answer. However, reward on problem solving process is difficult to define and parse. Therefore, we use the "critical value" or "critical words" to detect whether the model is on the right track. The critical value is a set of values or incomplete expression. The critical words are key concepts or terms. Both critical value and critical words are important to problem solving, non-trivial to guess, and never appearing in the question or final answer. You can refer to the following examples to understand the critical value and critical words.
-----------------------
## Example 1:

### Quesition: 
What are all values of $p$ such that for every $q>0$, we have $$\\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ Express your answer in interval notation in decimal form.
### Answer: 
First we'll simplify that complicated expression. We attempt to factor the numerator of the left side: \\begin{align*}\npq^2+p^2q+3q^2+3pq &= q(pq + p^2 + 3q + 3p) \\\\\n&= q[ p(q+p) + 3(q+p) ] \\\\\n&= q(p+3)(q+p).\n\\end{align*}Substituting this in for the numerator in our inequality gives $$\\frac{3q(p+3)(p+q)}{p+q}>2p^2q.$$We note that left hand side has $p+q$ in both the numerator and denominator. We can only cancel these terms if $p+q \\neq 0.$ Since we're looking for values of $p$ such that the inequality is true for all $q > 0,$ we need $p \\geq 0$ so that $p + q \\neq 0.$\nAlso because this must be true for every $q>0$, we can cancel the $q$'s on both sides. This gives \\begin{align*}\n3(p+3)&>2p^2\\Rightarrow\\\\\n3p+9&>2p^2 \\Rightarrow\\\\\n0&>2p^2-3p-9.\n\\end{align*}Now we must solve this quadratic inequality. We can factor the quadratic as $2p^2-3p-9=(2p+3)(p-3)$. The roots are $p=3$ and $p=-1.5$. Since a graph of this parabola would open upwards, we know that the value of $2p^2 - 3p - 9$ is negative between the roots, so the solution to our inequality is $-1.5<p<3.$ But we still need $0 \\leq p,$ so in interval notation the answer is $\\boxed{[0,3)}$.

### Critical Values and Words:
factor numerator <sep> q[ p(q+p) + 3(q+p) ] <sep> q(p + 3)(q + p) <sep> p + q numerator denominator <sep>  p + q \\neq  0 <sep> p \\geq 0 <sep> 2p^2 - 3p - 9 <sep> (2p + 3)(p - 3) <sep> root p = 3 p = -1.5 <sep> parabola open upward <sep> negative between roots

## Example 2:

### Quesition:
For all real numbers $r$ and $s$, define the mathematical operation $\\#$ such that the following conditions apply: $r\\ \\#\\ 0 = r, r\\ \\#\\ s = s\\ \\#\\ r$, and $(r + 1)\\ \\#\\ s = (r\\ \\#\\ s) + s + 1$. What is the value of $11\\ \\#\\ 5$?
### Answer:
Using the first two conditions, we have that $0 \\# 11 = 11 \\# 0 = 11.$\nUsing the third condition, with $r=0$ and $s=11$, we have that $1 \\# 11 = (0 \\# 11)+12=11+12.$\nAs we increase $r$ by $1$, we increase $r \\# 11$ by $s+1=11+1=12$. Since we want to increase $r$ $5$ times to find $11 \\#5 =5 \\# 11$, we want to increase $0 \\# 11$ by $12$ five times. Therefore, we have $11 \\# 5 = 5 \\# 11 = 11+ 5 \\cdot 12 = 11+60= \\boxed{71}.$\nMore generally,\n\\[a \\# b = ab + a + b.\\]
 
### Critical Values and Words:
0 \\# 11 <sep> 1 \\# 11 12 <sep> increase r by 1 <sep> r 1 \\# 11 12 <sep> 0 \\# 11 12 5 <sep> ab + a + b

## Example 3:

### Quesition:
To calculate $31^2$, Emily mentally figures the value $30^2$ and adds 61. Emily subtracts a number from $30^2$ to calculate $29^2$. What number does she subtract?
### Answer:
We see that $29^2 = (30 - 1)^2 = 30^2 - 2\\cdot 30 \\cdot 1 + 1 = 30^2 - 59$. Therefore, Emily subtracts $\\boxed{59}$.

### Critical Values and Words:
(30 - 1)^2 <sep> 2 30 1 + 1
-----------------------
You are given a math question and its answer. You need to extract the critical values and critical words from the answer with original notation, for example, using \\sqrt instead of sqrt or √.

### Question:"""


async def generate_questions(qa, prompt, result_list, exception_list, sem, client):
    async with sem:
        print(f"Extract critical value")
        tmp_user_prompt = prompt + qa["problem"] + "\n### Answer:\n" + qa["solution"]
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
                        "content": "### Critical Values and Words:\n"
                    }
                ],
                stream = False,
                temperature = 0.5,
            )
            # Read the response as json
            output = response.choices[0].message.content
            qa['process'] = output
            # Append the result to the result list
            result_list.append(qa)
        except Exception as e:
            print(f"Error in extracting cv")
            print(e)
            exception_list.append(qa)

    print(f"Finish extracting cv")


# A function to generate the dataset
async def generate_dataset(qa_dataset, prompt):
    # Initialize deepseek client
    client = AsyncOpenAI(api_key="sk-6a81291d661d46f9b4cd18f8cc43b693", base_url="https://api.deepseek.com")
    # Control the number of tasks running concurrently
    sem = asyncio.Semaphore(30)
    # Create a list to store the results
    result_list = []
    # Create a exception list to store the exception
    exception_list = []


    print("Start generating questions")
    # Create a list of tasks to generate questions for each vid_name
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
asyncio.run(generate_dataset(math_qa_dataset, prompt_cv_extraction_generation))