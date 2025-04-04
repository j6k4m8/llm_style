from openai import OpenAI

class PairedFormalityDataset:
    def __init__(self, api_key: str, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_test_example(self, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content
        return content


    def get_formality_pairs(self, formality_type: int, prompt: str):
        if formality_type not in [0, 1]:
            raise ValueError("Invalid value for 'formality_type'. It should be either 0 or 1.")

        # Define the system message based on formality type
        system_message = (
            "You will be given a sentence in informal style, and you need to provide the corresponding sentence in formal style."
            if formality_type == 0
            else "You will be given a sentence in formal style, and you need to provide the corresponding sentence in informal style."
        )
        system_message += (
            " The output should be a single sentence."
            " Please ensure that the sentence is clear and maintains the same meaning."
            " An example of the output format is: 'However, I do believe it to be punk', when the input is: 'I'd say it is punk though'."
            " Another example of the out format is: 'Gotta see both sides of the story', when the input is: 'You have to consider both sides of the story'."
        )

        # Call the OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        # Extract the response content
        content = completion.choices[0].message.content 

        # Create formality pairs based on the formality type
        if formality_type == 0:
            pairs = [(prompt, 0), (content, 1)] # 0 for informal, 1 for formal
        elif formality_type == 1:
            pairs = [(content, 0), (prompt, 1)]
        else:
            raise ValueError("Invalid values")

        return pairs
