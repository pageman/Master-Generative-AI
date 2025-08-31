
import { Stat, Module, Resource, AccordionItemData } from './types';

export const STATS_DATA: Stat[] = [
    { number: "5", label: "Progressive Modules" },
    { number: "3-6", label: "Months Duration" },
    { number: "50+", label: "Hands-on Projects" },
    { number: "100+", label: "Resources Curated" }
];

export const MODULES_DATA: Module[] = [
    {
        icon: "🌱",
        title: "Foundations",
        duration: "2-4 weeks",
        description: "Build mathematical, programming, and conceptual foundations necessary for generative AI.",
        skills: ["Python programming for AI", "Linear algebra and probability", "Basic neural network concepts", "Development environment setup"],
        difficulty: "Beginner",
        difficultyClass: "bg-green-100 text-green-800"
    },
    {
        icon: "🔧",
        title: "Core GenAI Concepts",
        duration: "3-5 weeks",
        description: "Master fundamental generative models including VAEs, attention mechanisms, and transformers.",
        skills: ["Variational Autoencoders (VAEs)", "Attention mechanisms", "Transformer architecture", "Dataset preprocessing"],
        difficulty: "Easy-Intermediate",
        difficultyClass: "bg-orange-100 text-orange-800"
    },
    {
        icon: "⚙️",
        title: "Intermediate Models",
        duration: "4-6 weeks",
        description: "Explore GANs, flow models, and evaluation techniques for generative systems.",
        skills: ["Generative Adversarial Networks", "Flow-based models", "Energy-based models", "Evaluation metrics (FID, IS)"],
        difficulty: "Intermediate",
        difficultyClass: "bg-orange-100 text-orange-800"
    },
    {
        icon: "🎯",
        title: "Advanced Applications",
        duration: "4-6 weeks",
        description: "Master diffusion models, LLMs, and multimodal generation for real-world applications.",
        skills: ["Diffusion models (DDPM, Stable Diffusion)", "Large Language Models", "Multimodal generation", "Fine-tuning and adaptation"],
        difficulty: "Intermediate-Advanced",
        difficultyClass: "bg-red-100 text-red-800"
    },
    {
        icon: "🚀",
        title: "Research & Ethics",
        duration: "4-8 weeks",
        description: "Engage with cutting-edge research, ethical considerations, and production-scale systems.",
        skills: ["Advanced architectures", "Ethical AI and safety", "Research frontiers", "Production MLOps systems"],
        difficulty: "Expert",
        difficultyClass: "bg-red-100 text-red-800"
    }
];

export const RESOURCES_DATA: Resource[] = [
    { type: "📚 Courses", title: "Curated Learning Paths", description: "Access to premium courses from Stanford, Berkeley, DeepLearning.AI, and industry leaders." },
    { type: "📄 Papers", title: "Research Papers", description: "Essential papers from foundational VAEs to cutting-edge diffusion models and LLMs." },
    { type: "🛠️ Tools", title: "Development Tools", description: "PyTorch, Hugging Face, Stable Diffusion, and other essential tools for GenAI development." },
    { type: "💼 Projects", title: "Hands-On Projects", description: "Progressive projects from simple text generators to production-ready multimodal applications." },
    { type: "🌐 Community", title: "Learning Community", description: "Connect with fellow learners, researchers, and industry practitioners in active communities." },
    { type: "📊 Assessment", title: "Progress Tracking", description: "Milestone checklists, self-assessments, and portfolio development guidance." }
];

export const ACCORDION_DATA: AccordionItemData[] = [
  {
    icon: "🌱",
    title: "Module 1: Foundations & Prerequisites",
    learningObjectives: "Build the mathematical, programming, and conceptual foundation necessary for generative AI without overwhelming beginners.",
    keyTopics: [
      "Conceptual Foundation: GenAI vs Discriminative AI, ML paradigms, AI terminology",
      "Mathematical Prerequisites: Linear algebra, probability, calculus essentials",
      "Programming Foundation: Python for AI, NumPy, Pandas, Matplotlib",
      "Development Setup: Jupyter Notebooks, Google Colab, version control",
    ],
    projects: [
      {
        title: "Set up development environment",
        description: "Before you start, make sure you have Python, a virtual environment tool, and a code editor installed.",
        codeId: "m1p1", codeLang: "", code: "",
      },
      {
        title: "Create data visualizations with sample datasets",
        description: "Explore a dataset to understand its structure and patterns.",
        codeId: "m1p2", codeLang: "Python",
        code: `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# A simple dataset for demonstration
data = {
    'x': np.random.rand(50) * 10,
    'y': np.random.rand(50) * 10,
    'category': np.random.choice(['A', 'B', 'C'], 50)
}
df = pd.DataFrame(data)

# Goal: Create a scatter plot to visualize the data
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', hue='category', data=df, s=100)
plt.title('Sample Data Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Category')
plt.show()

print("Visualization complete.")
`,
      },
      {
        title: "Build a simple random data generator",
        description: "Practice generating various types of random data to mimic real-world inputs.",
        codeId: "m1p3", codeLang: "Python",
        code: `
import numpy as np
import string
import random

def generate_random_data(num_samples):
    """Generates random data with different types."""
    
    # Generate random integers between 1 and 100
    int_data = np.random.randint(1, 101, size=num_samples)
    
    # Generate random floating-point numbers from a normal distribution
    float_data = np.random.randn(num_samples)
    
    # Generate random strings
    str_data = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(num_samples)]
    
    return int_data, float_data, str_data

# Example usage
num_samples = 10
int_samples, float_samples, str_samples = generate_random_data(num_samples)

print("Generated Integer Data:", int_samples)
print("Generated Float Data:", float_samples)
print("Generated String Data:", str_samples)
`,
      },
    ],
  },
  {
    icon: "🔧",
    title: "Module 2: Core GenAI Concepts",
    learningObjectives: "Understand fundamental generative models and techniques, building from simple probabilistic models to sophisticated approaches.",
    keyTopics: [
      "Basic Generative Models: Autoregressive models, Markov chains, MLE",
      "Variational Autoencoders: Encoder-decoder architecture, latent space, reparameterization",
      "Attention & Transformers: Self-attention, transformer architecture, positional encoding",
      "Dataset Management: Common datasets, preprocessing, augmentation",
    ],
    projects: [
      {
        title: "Week 1: Markov Chain Text Generator",
        description: "Create a model that generates new text by predicting the next word based on the previous word.",
        codeId: "m2w1", codeLang: "Python",
        code: `
import random

class MarkovChain:
    def __init__(self):
        self.transition_matrix = {}
    
    def train(self, text):
        words = text.split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i+1]
            if current_word not in self.transition_matrix:
                self.transition_matrix[current_word] = []
            self.transition_matrix[current_word].append(next_word)
    
    def generate_text(self, start_word, num_words):
        generated_words = [start_word]
        current_word = start_word
        for _ in range(num_words - 1):
            if current_word in self.transition_matrix:
                next_word = random.choice(self.transition_matrix[current_word])
                generated_words.append(next_word)
                current_word = next_word
            else:
                break
        return ' '.join(generated_words)

# Example usage
text = "the quick brown fox jumps over the lazy dog"
mc = MarkovChain()
mc.train(text)
generated_text = mc.generate_text("the", 5)
print("Generated Text:", generated_text)
`,
      },
      {
        title: "Week 2: Basic VAE on MNIST",
        description: "Implement a Variational Autoencoder to generate new images of handwritten digits.",
        codeId: "m2w2", codeLang: "PyTorch",
        code: `
import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

print("VAE model template created.")
`,
      },
    ],
  },
  {
      icon: '⚙️',
      title: 'Module 3: Intermediate Models',
      learningObjectives: 'Explore GANs, flow models, and evaluation techniques for generative systems.',
      keyTopics: [
          'Generative Adversarial Networks: DCGAN, training dynamics, mode collapse',
          'Improved GANs: Wasserstein GAN, Gradient Penalty (WGAN-GP)',
          'Conditional GANs: Controllable image synthesis',
          'Flow-based models: RealNVP, invertible neural networks, explicit likelihood',
      ],
      projects: [
          {
              title: 'Week 1-2: DCGAN for Image Generation',
              description: 'Build a Generative Adversarial Network that uses convolutional layers to generate images.',
              codeId: 'm3w1',
              codeLang: 'PyTorch',
              code: `
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8), nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4), nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2), nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input): return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input): return self.main(input)

print("DCGAN model template created.")
`,
          },
      ],
  },
  {
      icon: '🎯',
      title: 'Module 4: Advanced Models & Real-World Applications',
      learningObjectives: 'Master state-of-the-art generative models and their applications in diverse domains with a focus on practical deployment.',
      keyTopics: [
          'Diffusion Models: DDPM, Stable Diffusion, score-based models, guidance techniques',
          'Large Language Models: GPT architecture, text generation, prompt engineering',
          'Multimodal Generation: Text-to-image, vision-language understanding',
          'Fine-tuning: Transfer learning, LoRA, parameter-efficient methods',
      ],
      projects: [
          {
              title: 'Week 3-4: Fine-tune a pre-trained LLM (e.g., GPT-2)',
              description: 'Adapt a powerful pre-trained model for your own text generation task.',
              codeId: 'm4w3',
              codeLang: 'Hugging Face',
              code: `
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "GenAI is an exciting field with many applications.",
]

encodings = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
input_ids = encodings['input_ids']
labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100

print("LLM fine-tuning template created.")
`,
          },
      ],
  },
  {
      icon: '🚀',
      title: 'Module 5: Research & Ethics',
      learningObjectives: 'Engage with current research frontiers, address ethical considerations, and implement complex generative AI systems at scale.',
      keyTopics: [
          'Advanced Architectures: Score-based models, hybrid approaches, controllable generation',
          'Scaling & Optimization: Large-scale training, model compression, distributed systems',
          'Ethical AI: Bias detection, safety frameworks, responsible deployment',
          'Production Systems: MLOps, API design, monitoring, cost optimization',
      ],
      projects: [
          {
              title: 'Week 6-8: Capstone Project: Build a Full-Stack GenAI Application',
              description: 'Build a complete web application that showcases one of your generative models.',
              codeId: 'm5w6',
              codeLang: 'Flask/Python',
              code: `
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Assume 'model' is your loaded generative model
# model = YourGenerativeModel()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

@app.route('/generate', methods=['POST'])
def generate_content():
    try:
        data = request.json
        input_text = data.get('prompt')
        
        # This is a placeholder for your model's inference logic
        # output_tensor = model(input_text)
        # output_text = convert_tensor_to_text(output_tensor)
        output_text = "Generated content for: " + input_text

        return jsonify({"result": output_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the GenAI Capstone Backend!"

if __name__ == '__main__':
    app.run(debug=True)

print("Full-stack capstone template created.")
`,
          },
      ],
  }
];
