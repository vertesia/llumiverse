/**
 * Example demonstrating batch inference for Vertex AI and Bedrock
 */

import { Modalities, PromptRole, PromptSegment } from '@llumiverse/core';
import { BedrockDriver } from '@llumiverse/drivers/bedrock';
import { VertexAIDriver } from '@llumiverse/drivers/vertexai';

/**
 * Example using Vertex AI batch inference
 */
async function vertexAIBatchExample() {
  console.log('Running Vertex AI Batch Inference Example');

  // Create a Vertex AI driver (configure with your project and region)
  const driver = new VertexAIDriver({
    project: process.env.VERTEX_PROJECT_ID || 'your-gcp-project-id',
    region: process.env.VERTEX_REGION || 'us-central1'
  });

  // Prepare multiple prompts for batch processing
  const batchInputs = [];

  // Add several sample prompts
  for (let i = 0; i < 5; i++) {
    const segments: PromptSegment[] = [
      {
        role: PromptRole.user,
        content: `Tell me an interesting fact about planet ${i+1} in our solar system.`
      }
    ];

    batchInputs.push({
      segments,
      options: {
        model: 'gemini-1.0-pro',
        output_modality: Modalities.text,
        batch_id: `item-${i}`  // Assign a unique ID to each item
      }
    });
  }

  try {
    // Start the batch inference job
    console.log('Starting batch inference job...');
    const batchJob = await driver.startBatchInference(batchInputs, {
      // Optional batch options
      maxWorkerCount: 2,
      machineType: 'e2-standard-4'
    });

    console.log('Batch job created:', batchJob);
    console.log(`Job ID: ${batchJob.id}`);

    // Poll for job status
    let jobStatus = batchJob;
    let complete = false;
    
    console.log('Polling for job status...');
    while (!complete) {
      await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds
      
      jobStatus = await driver.getBatchInferenceJob(batchJob.id);
      console.log(`Job status: ${jobStatus.status}`);
      
      if (jobStatus.status === 'succeeded' || 
          jobStatus.status === 'failed' || 
          jobStatus.status === 'cancelled' ||
          jobStatus.status === 'partial') {
        complete = true;
      }
    }

    // Get results if job completed successfully
    if (jobStatus.status === 'succeeded' || jobStatus.status === 'partial') {
      console.log('Retrieving batch results...');
      const results = await driver.getBatchInferenceResults(batchJob.id);
      
      console.log('Batch Results:');
      results.results.forEach(item => {
        console.log(`\nItem ID: ${item.item_id}`);
        console.log(`Result: ${item.result}`);
        if (item.token_usage) {
          console.log(`Token usage: ${JSON.stringify(item.token_usage)}`);
        }
      });
    } else {
      console.log('Job did not complete successfully');
      console.log(`Error details: ${jobStatus.details}`);
    }
  } catch (error) {
    console.error('Error in batch processing:', error);
  }
}

/**
 * Example using Bedrock batch inference
 */
async function bedrockBatchExample() {
  console.log('Running AWS Bedrock Batch Inference Example');

  // Create a Bedrock driver (configure with your AWS region)
  const driver = new BedrockDriver({
    region: process.env.AWS_REGION || 'us-east-1',
    training_bucket: process.env.AWS_S3_BUCKET || 'your-s3-bucket',
    training_role_arn: process.env.AWS_ROLE_ARN || 'your-role-arn'
  });

  // Prepare multiple prompts for batch processing
  const batchInputs = [];

  // Add several sample prompts
  for (let i = 0; i < 5; i++) {
    const segments: PromptSegment[] = [
      {
        role: PromptRole.system,
        content: 'You are a helpful assistant.'
      },
      {
        role: PromptRole.user,
        content: `Tell me an interesting fact about animal number ${i+1} on the endangered species list.`
      }
    ];

    batchInputs.push({
      segments,
      options: {
        model: 'anthropic.claude-3-sonnet-20240229-v1:0',
        output_modality: Modalities.text,
        batch_id: `item-${i}`  // Assign a unique ID to each item
      }
    });
  }

  try {
    // Start the batch inference job
    console.log('Starting batch inference job...');
    const batchJob = await driver.startBatchInference(batchInputs, {
      // Optional batch options
      maxConcurrentInvocations: 2,
      timeoutInSeconds: 3600
    });

    console.log('Batch job created:', batchJob);
    console.log(`Job ID: ${batchJob.id}`);

    // Poll for job status
    let jobStatus = batchJob;
    let complete = false;
    
    console.log('Polling for job status...');
    while (!complete) {
      await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds
      
      jobStatus = await driver.getBatchInferenceJob(batchJob.id);
      console.log(`Job status: ${jobStatus.status}`);
      
      if (jobStatus.status === 'succeeded' || 
          jobStatus.status === 'failed' || 
          jobStatus.status === 'cancelled' ||
          jobStatus.status === 'partial') {
        complete = true;
      }
    }

    // Get results if job completed successfully
    if (jobStatus.status === 'succeeded' || jobStatus.status === 'partial') {
      console.log('Retrieving batch results...');
      const results = await driver.getBatchInferenceResults(batchJob.id);
      
      console.log('Batch Results:');
      results.results.forEach(item => {
        console.log(`\nItem ID: ${item.item_id}`);
        console.log(`Result: ${item.result}`);
        if (item.token_usage) {
          console.log(`Token usage: ${JSON.stringify(item.token_usage)}`);
        }
      });
    } else {
      console.log('Job did not complete successfully');
      console.log(`Error details: ${jobStatus.details}`);
    }
  } catch (error) {
    console.error('Error in batch processing:', error);
  }
}

/**
 * Example using OpenAI batch inference
 */
async function openAIBatchExample() {
  console.log('Running OpenAI Batch Inference Example');

  // Import the OpenAI driver
  const { OpenAIDriver } = await import('@llumiverse/drivers/openai');

  // Create an OpenAI driver (configure with your API key)
  const driver = new OpenAIDriver({
    apiKey: process.env.OPENAI_API_KEY || 'your-openai-api-key'
  });

  // Prepare multiple prompts for batch processing
  const batchInputs = [];

  // Add several sample prompts
  for (let i = 0; i < 5; i++) {
    const segments: PromptSegment[] = [
      {
        role: PromptRole.system,
        content: 'You are a helpful assistant.'
      },
      {
        role: PromptRole.user,
        content: `Give me a short summary of book ${i+1} in the Harry Potter series.`
      }
    ];

    batchInputs.push({
      segments,
      options: {
        model: 'gpt-3.5-turbo',
        output_modality: Modalities.text,
        batch_id: `item-${i}`  // Assign a unique ID to each item
      }
    });
  }

  try {
    // Start the batch inference job
    console.log('Starting batch inference job...');
    const batchJob = await driver.startBatchInference(batchInputs, {
      // Optional batch options
      concurrency: 2,
      timeout: 60000,
      retryCount: 2
    });

    console.log('Batch job created:', batchJob);
    console.log(`Job ID: ${batchJob.id}`);

    // Poll for job status
    let jobStatus = driver.getBatchInferenceJob(batchJob.id);
    let complete = false;
    
    console.log('Polling for job status...');
    while (!complete) {
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
      
      jobStatus = driver.getBatchInferenceJob(batchJob.id);
      console.log(`Job status: ${jobStatus.status}`);
      
      if (jobStatus.status === 'succeeded' || 
          jobStatus.status === 'failed' || 
          jobStatus.status === 'cancelled' ||
          jobStatus.status === 'partial') {
        complete = true;
      }
    }

    // Get results if job completed successfully
    if (jobStatus.status === 'succeeded' || jobStatus.status === 'partial') {
      console.log('Retrieving batch results...');
      const results = driver.getBatchInferenceResults(batchJob.id);
      
      console.log('Batch Results:');
      results.results.forEach(item => {
        console.log(`\nItem ID: ${item.item_id}`);
        console.log(`Result: ${item.result}`);
        if (item.token_usage) {
          console.log(`Token usage: ${JSON.stringify(item.token_usage)}`);
        }
      });
    } else {
      console.log('Job did not complete successfully');
      console.log(`Error details: ${jobStatus.details}`);
    }
  } catch (error) {
    console.error('Error in batch processing:', error);
  }
}

// Run examples
async function main() {
  // Choose which example to run based on environment
  if (process.env.EXAMPLE_PROVIDER === 'vertex') {
    await vertexAIBatchExample();
  } else if (process.env.EXAMPLE_PROVIDER === 'bedrock') {
    await bedrockBatchExample();
  } else if (process.env.EXAMPLE_PROVIDER === 'openai') {
    await openAIBatchExample();
  } else {
    console.log('Set EXAMPLE_PROVIDER environment variable to "vertex", "bedrock", or "openai" to run examples');
    console.log('For example: EXAMPLE_PROVIDER=openai node batch-inference.js');
  }
}

main().catch(console.error);