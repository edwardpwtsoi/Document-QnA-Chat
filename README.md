# Document Q&A Chatbot

This chatbot answers questions about documents using a combination of local data and web search capabilities.

## Features

- Processes CSV data files
- Extracts information from PDF documents
- Performs web searches when local data is insufficient
- Provides accurate and sourced responses to queries

## Architecture

![Architecture Diagram](architecture.png)

The chatbot uses a Retrieval Augmented Generation (RAG) approach with:

1. **Data Processing**: Loading CSV and PDF data into a vector database
2. **Retrieval**: Finding relevant information from the vector database or web search
3. **Reasoning**: Using an LLM to generate accurate responses
4. **Interface**: Streamlit web interface for interaction

## Setup Instructions

### Prerequisites

- Python 3.10 or 3.11 (Python 3.12 not yet fully supported)
- AWS Account with Bedrock access
- AWS credentials configured with the following permissions:
  - `bedrock:InvokeModel`
  - `bedrock:ListFoundationModels`
  - `bedrock:GetFoundationModel`
  - Access to Claude v2 and Titan Embeddings models in Bedrock

### AWS Setup

1. Create an AWS account if you don't have one
2. Enable Amazon Bedrock in your AWS account
3. Create an IAM user with Bedrock access and attach a policy with the required permissions:

   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "bedrock:InvokeModel",
                   "bedrock:ListFoundationModels",
                   "bedrock:GetFoundationModel"
               ],
               "Resource": "*"
           }
       ]
   }
   ```

4. Configure AWS credentials using one of these methods:
   - Through environment variables:

     ```bash
     export AWS_ACCESS_KEY_ID=your_access_key
     export AWS_SECRET_ACCESS_KEY=your_secret_key
     export AWS_DEFAULT_REGION=your_region  # e.g., us-east-1
     ```

   - Through a .env file:

     ```text
     AWS_ACCESS_KEY_ID=your_access_key
     AWS_SECRET_ACCESS_KEY=your_secret_key
     AWS_DEFAULT_REGION=your_region
     ```

   - Through AWS SSO (recommended):
     1. Configure AWS SSO in your AWS Organization
     2. Install AWS CLI v2
     3. Configure SSO with AWS CLI:

        ```bash
        aws configure sso
        ```

     4. Follow the prompts to:
        - Enter your SSO start URL
        - Enter your SSO Region
        - Choose your SSO account and role
        - Name your SSO profile (e.g., 'document-qa')
     5. Set the profile in your environment:

        ```bash
        export AWS_PROFILE=document-qa
        ```

     6. Verify SSO login:

        ```bash
        aws sso login --profile document-qa
        ```

### Installation

1. Clone this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your AWS credentials as described above

### Running Locally

```bash
streamlit run app.py -- --domain <DOCUMENT DOMAIN>
```
