#!/usr/bin/env python
"""
Simple script to prepare a math tutoring dataset
"""

import os
import json
import random
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Math tutoring examples
MATH_EXAMPLES = [
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: How do I solve quadratic equations?",
        "response": "To solve a quadratic equation (ax² + bx + c = 0):\n\n1. Try factoring if possible: Find numbers that multiply to c and add to b\n2. Use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a\n3. Complete the square for specific cases\n\nExample: For x² - 5x + 6 = 0\nFactoring: (x-2)(x-3) = 0\nSo x = 2 or x = 3\n\nAlways check your answers by substituting back into the original equation."
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: What is the derivative of sin(x)?",
        "response": "The derivative of sin(x) is cos(x).\n\nThis is one of the fundamental derivatives in calculus. You can prove this using the limit definition of a derivative:\n\nf'(x) = lim[h→0] [f(x+h) - f(x)]/h\n\nFor sin(x), working through this limit with trigonometric identities leads to cos(x).\n\nSimilarly, the derivative of cos(x) is -sin(x).\n\nThese derivatives are essential for solving problems involving oscillations, waves, and periodic phenomena."
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: How do I find the area of a circle?",
        "response": "The area of a circle is given by the formula:\n\nArea = πr²\n\nWhere r is the radius of the circle and π (pi) is approximately 3.14159.\n\nSteps to find the area:\n1. Identify the radius of the circle\n2. Square the radius (multiply it by itself)\n3. Multiply by π\n\nExample: If a circle has radius 5 cm, the area would be:\nArea = π × 5² = π × 25 = 78.54 cm²\n\nIf you only know the diameter (d), remember that radius = diameter/2, so the formula becomes:\nArea = π(d/2)² = πd²/4"
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: Explain how to solve systems of linear equations.",
        "response": "Systems of linear equations can be solved using several methods:\n\n1. Substitution Method:\n   - Solve one equation for one variable\n   - Substitute that expression into the other equation(s)\n   - Solve for the remaining variable\n   - Back-substitute to find the other variable\n\n2. Elimination Method:\n   - Manipulate equations so a variable has the same coefficient with opposite signs\n   - Add the equations to eliminate that variable\n   - Solve for the remaining variable\n   - Back-substitute\n\n3. Matrix Methods:\n   - Set up an augmented matrix\n   - Use row operations to get row-echelon form\n   - Back-substitute to find all variables\n\nExample: Solve { 2x + y = 5\n             { x - y = 1\n\nUsing elimination: Add the equations to get 3x = 6, so x = 2\nThen y = 5 - 2(2) = 1\n\nSolution: (2, 1)"
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: What is the Pythagorean theorem and how do I use it?",
        "response": "The Pythagorean theorem states that in any right triangle, the square of the length of the hypotenuse (the side opposite the right angle) equals the sum of squares of the other two sides.\n\nMathematically: a² + b² = c²\nWhere c is the hypotenuse, and a and b are the other two sides.\n\nHow to use it:\n1. Identify the right triangle and its sides\n2. Determine which side you need to find\n3. Substitute the known values into the formula\n4. Solve for the unknown side\n\nExample: If a = 3 and b = 4, find c:\nc² = 3² + 4² = 9 + 16 = 25\nc = √25 = 5\n\nYou can also use it to determine if a triangle is a right triangle by checking if the equation holds true."
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: How do I calculate probability?",
        "response": "Basic probability is calculated as:\n\nP(event) = Number of favorable outcomes / Total number of possible outcomes\n\nKey probability concepts:\n\n1. For independent events, multiply probabilities:\n   P(A and B) = P(A) × P(B)\n\n2. For mutually exclusive events, add probabilities:\n   P(A or B) = P(A) + P(B)\n\n3. For non-mutually exclusive events:\n   P(A or B) = P(A) + P(B) - P(A and B)\n\n4. Complement rule:\n   P(not A) = 1 - P(A)\n\nExample: In a standard deck of 52 cards, the probability of drawing a heart is:\nP(heart) = 13/52 = 1/4 = 0.25 or 25%\n\nProbability always ranges from 0 (impossible) to 1 (certain)."
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: What are the rules for exponents?",
        "response": "Key rules for exponents:\n\n1. Product rule: x^a × x^b = x^(a+b)\n   Example: 2^3 × 2^4 = 2^7 = 128\n\n2. Quotient rule: x^a ÷ x^b = x^(a-b)\n   Example: 2^5 ÷ 2^2 = 2^3 = 8\n\n3. Power of a power: (x^a)^b = x^(a×b)\n   Example: (2^3)^2 = 2^6 = 64\n\n4. Power of a product: (x×y)^a = x^a × y^a\n   Example: (2×3)^2 = 2^2 × 3^2 = 4 × 9 = 36\n\n5. Power of a quotient: (x÷y)^a = x^a ÷ y^a\n   Example: (4÷2)^3 = 4^3 ÷ 2^3 = 64 ÷ 8 = 8\n\n6. Negative exponent: x^(-a) = 1/(x^a)\n   Example: 2^(-3) = 1/(2^3) = 1/8 = 0.125\n\n7. Zero exponent: x^0 = 1 (if x ≠ 0)\n   Example: 7^0 = 1\n\n8. Fractional exponent: x^(m/n) = ⁿ√(x^m)\n   Example: 8^(2/3) = ∛(8^2) = ∛64 = 4"
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: How do I calculate limits in calculus?",
        "response": "Strategies for calculating limits:\n\n1. Direct substitution: If the function is continuous at point a, simply substitute x = a\n   lim[x→2] (x² + 3x) = 2² + 3(2) = 4 + 6 = 10\n\n2. Factoring: For limits with divisions that give 0/0\n   lim[x→3] (x²-9)/(x-3) = lim[x→3] (x+3)(x-3)/(x-3) = lim[x→3] (x+3) = 6\n\n3. Rationalization: For limits with square roots\n   lim[x→4] (√x-2)/(x-4) = lim[x→4] (√x-2)/((x-4)(√x+2)) = lim[x→4] 1/(√x+2) = 1/4\n\n4. L'Hôpital's Rule: For indeterminate forms like 0/0 or ∞/∞, take derivatives of numerator and denominator\n   lim[x→0] sin(x)/x = lim[x→0] cos(x)/1 = 1\n\n5. Special limits:\n   lim[x→0] sin(x)/x = 1\n   lim[x→∞] (1+1/x)^x = e\n\nAlways look for indeterminate forms (0/0, ∞/∞, etc.) and apply the appropriate technique."
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: What are the laws of logarithms?",
        "response": "The fundamental laws of logarithms:\n\n1. Product rule: log_b(MN) = log_b(M) + log_b(N)\n   Example: log(2×8) = log(2) + log(8)\n\n2. Quotient rule: log_b(M/N) = log_b(M) - log_b(N)\n   Example: log(50/5) = log(50) - log(5)\n\n3. Power rule: log_b(M^p) = p × log_b(M)\n   Example: log(3^4) = 4 × log(3)\n\n4. Base change formula: log_b(x) = log_a(x) / log_a(b)\n   Example: log_2(16) = log(16) / log(2)\n\n5. Special properties:\n   - log_b(1) = 0 (for any base b)\n   - log_b(b) = 1 (for any base b)\n   - log_b(b^x) = x\n   - b^(log_b(x)) = x\n\nThese rules help simplify complex logarithmic expressions and solve logarithmic equations."
    },
    {
        "instruction": "You are a helpful and knowledgeable math tutor. Help the student with their question: How do I approach word problems?",
        "response": "Systematic approach to math word problems:\n\n1. Read carefully: Understand what's being asked\n   - Identify the unknown (what you need to find)\n   - Note all given information\n   - Look for keywords that suggest operations\n\n2. Define variables: Assign symbols to unknowns\n   - Use meaningful variables (like t for time)\n   - Write down what each variable represents\n\n3. Set up equations: Translate the problem into mathematical form\n   - Use the relationships described in the problem\n   - Check units for consistency\n\n4. Solve: Use appropriate mathematical techniques\n   - Apply algebra, arithmetic, etc.\n   - Keep track of steps\n\n5. Check: Verify your solution\n   - Substitute back into the original problem\n   - Ask if the answer makes sense\n   - Check units\n\nExample: \"A train travels 240 miles at a constant speed. If the speed had been 10 mph faster, the trip would have taken 1 hour less. Find the speed.\"\n\n1. Define: Let s = speed in mph\n2. Time = distance/speed, so original time = 240/s\n3. New time = 240/(s+10) = (240/s) - 1\n4. Set up equation: 240/(s+10) = 240/s - 1\n5. Solve: s = 60 mph\n6. Check: At 60 mph, time is 240/60 = 4 hours; at 70 mph, time is 240/70 ≈ 3.43 hours (difference is about 1 hour)"
    }
]

def main():
    # Settings
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DATA_DIR = Path("data/math_tutoring")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loaded tokenizer for {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Fallback to local tokenizer
        model_dir = Path("checkpoints") / MODEL_NAME.replace("/", "-")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        print(f"Loaded tokenizer from local directory")
    
    # Augment the dataset by creating variations
    augmented_examples = []
    for example in MATH_EXAMPLES:
        augmented_examples.append(example)
        
        # Create variations with different phrasings
        instruction = example["instruction"]
        response = example["response"]
        
        # Extract the question part
        question_part = instruction.split("How do I ")[1] if "How do I " in instruction else instruction.split("Help the student with their question: ")[1]
        
        # Create variations
        variations = [
            f"You are a helpful and knowledgeable math tutor. Explain {question_part}",
            f"You are a helpful and knowledgeable math tutor. How would you solve {question_part}",
            f"You are a helpful and knowledgeable math tutor. The student asks: {question_part}"
        ]
        
        for variation in variations:
            augmented_examples.append({
                "instruction": variation,
                "response": response
            })
    
    print(f"Generated {len(augmented_examples)} examples")
    
    # Split into train and validation
    random.shuffle(augmented_examples)
    split_idx = int(0.9 * len(augmented_examples))
    train_examples = augmented_examples[:split_idx]
    val_examples = augmented_examples[split_idx:]
    
    # Tokenize and save examples
    train_data = []
    for ex in train_examples:
        input_ids = tokenizer.encode(ex["instruction"])
        response_ids = tokenizer.encode(ex["response"])
        
        train_data.append({
            "input_ids": input_ids,
            "labels": response_ids
        })
    
    val_data = []
    for ex in val_examples:
        input_ids = tokenizer.encode(ex["instruction"])
        response_ids = tokenizer.encode(ex["response"])
        
        val_data.append({
            "input_ids": input_ids,
            "labels": response_ids
        })
    
    # Save tokenized data
    torch.save(train_data, DATA_DIR / "train.pt")
    torch.save(val_data, DATA_DIR / "test.pt")
    
    # Save raw examples for reference
    with open(DATA_DIR / "train_examples.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    with open(DATA_DIR / "test_examples.json", "w") as f:
        json.dump(val_examples, f, indent=2)
    
    print(f"Dataset prepared and saved to {DATA_DIR}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

if __name__ == "__main__":
    main()