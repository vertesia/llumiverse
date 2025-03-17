import { BatchInferenceJob, BatchInferenceJobStatus, BatchInferenceResult, BatchItemResult, ExecutionOptions, PromptSegment } from '@llumiverse/core';
import { v4 as uuidv4 } from 'uuid';
import { BaseOpenAIDriver } from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { convertRoles, insert_image_detail, isNonStructureSupporting } from '../index.js';

/**
 * Options for OpenAI batch inference
 */
export interface OpenAIBatchOptions {
  /**
   * Maximum number of concurrent API requests
   * Defaults to 5
   */
  concurrency?: number;

  /**
   * Timeout in milliseconds for each individual request
   * Default: 120000 (2 minutes)
   */
  timeout?: number;

  /**
   * Chunk size for batch processing
   * Default: 20
   */
  batchSize?: number;

  /**
   * Retry count for failed requests
   * Default: 3
   */
  retryCount?: number;

  /**
   * Retry delay in milliseconds
   * Default: 1000 (1 second)
   */
  retryDelay?: number;
}

/**
 * Prepares inputs for batch processing
 */
async function prepareInputs(
  driver: BaseOpenAIDriver,
  batchInputs: { segments: PromptSegment[], options: ExecutionOptions }[]
): Promise<{ id: string, options: ExecutionOptions, prompt: any }[]> {
  const preparedInputs = [];
  
  for (const input of batchInputs) {
    try {
      const prompt = await driver.createPrompt(input.segments, input.options);
      const itemId = input.options.batch_id || uuidv4();
      
      // Prepare input with ID, options and prompt
      preparedInputs.push({
        id: itemId,
        options: input.options,
        prompt
      });
    } catch (error) {
      driver.logger.error(`Error preparing batch input: ${error}`);
    }
  }
  
  return preparedInputs;
}

/**
 * Process a chunk of the batch inputs
 */
async function processChunk(
  driver: BaseOpenAIDriver, 
  chunk: { id: string, options: ExecutionOptions, prompt: any }[],
  retryCount: number,
  retryDelay: number,
  timeout: number
): Promise<BatchItemResult[]> {
  const results: BatchItemResult[] = [];
  
  // Process each item in the chunk with retries
  await Promise.all(chunk.map(async (item) => {
    let attempts = 0;
    let success = false;
    let result: BatchItemResult = {
      item_id: item.id,
      error: 'Maximum retries exceeded'
    };
    
    while (attempts < retryCount && !success) {
      try {
        // Set up AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        // Process model options
        const model_options = item.options.model_options as any;
        convertRoles(item.prompt, item.options.model);
        insert_image_detail(item.prompt, model_options?.image_detail ?? "auto");
        
        const useTools = !isNonStructureSupporting(item.options.model);
        
        // Make the API request
        const response = await driver.service.chat.completions.create({
          model: item.options.model,
          messages: item.prompt,
          reasoning_effort: model_options?.reasoning_effort,
          temperature: model_options?.temperature,
          top_p: model_options?.top_p,
          presence_penalty: model_options?.presence_penalty,
          frequency_penalty: model_options?.frequency_penalty,
          n: 1,
          max_completion_tokens: model_options?.max_tokens,
          tools: useTools && item.options.result_schema && driver.provider.includes("openai")
            ? [
                {
                  function: {
                    name: "format_output",
                    parameters: item.options.result_schema as any,
                  },
                  type: "function"
                }
              ]
            : undefined,
          tool_choice: useTools && item.options.result_schema
            ? {
                type: 'function',
                function: { name: "format_output" }
              } 
            : undefined,
        }, { signal: controller.signal });
        
        clearTimeout(timeoutId);
        
        // Extract relevant data
        const choice = response.choices[0];
        let content;
        
        if (useTools && driver.provider.includes("openai") && item.options.result_schema) {
          content = choice?.message.tool_calls?.[0].function.arguments ?? choice.message.content ?? '';
        } else {
          content = choice.message.content ?? '';
        }
        
        // Create result object
        result = {
          item_id: item.id,
          result: content,
          token_usage: {
            prompt: response.usage?.prompt_tokens,
            result: response.usage?.completion_tokens,
            total: response.usage?.total_tokens
          }
        };
        
        success = true;
      } catch (error) {
        attempts++;
        if (attempts < retryCount) {
          await new Promise(resolve => setTimeout(resolve, retryDelay));
        } else {
          result.error = error instanceof Error ? error.message : 'Unknown error occurred';
        }
      }
    }
    
    results.push(result);
  }));
  
  return results;
}

/**
 * Implementation of batch inference for OpenAI
 */
export async function startBatchInference(
  driver: BaseOpenAIDriver,
  batchInputs: { segments: PromptSegment[], options: ExecutionOptions }[],
  batchOptions: OpenAIBatchOptions = {}
): Promise<BatchInferenceJob> {
  if (!batchInputs || batchInputs.length === 0) {
    throw new Error('Batch inputs cannot be empty');
  }

  // Use the model from the first input as the model for the batch job
  const model = batchInputs[0].options.model;
  if (!model) {
    throw new Error('Model must be specified in the first input options');
  }

  const jobId = `openai-batch-${uuidv4()}`;
  const timestamp = new Date();
  
  // Set up options with defaults
  const concurrency = batchOptions.concurrency || 5;
  const batchSize = batchOptions.batchSize || 20;
  const retryCount = batchOptions.retryCount || 3;
  const retryDelay = batchOptions.retryDelay || 1000;
  const timeout = batchOptions.timeout || 120000;
  
  // Create the batch job directory for results
  const batchDir = path.join(os.tmpdir(), 'llumiverse-openai-batch', jobId);
  if (!fs.existsSync(batchDir)) {
    fs.mkdirSync(batchDir, { recursive: true });
  }
  
  // Create batch job record file
  const batchJobFile = path.join(batchDir, 'job.json');
  const batchJob: BatchInferenceJob = {
    id: jobId,
    status: BatchInferenceJobStatus.created,
    model,
    input_count: batchInputs.length,
    start_time: timestamp,
    details: `Batch job created with ${batchInputs.length} inputs`,
  };
  
  // Save job info
  fs.writeFileSync(batchJobFile, JSON.stringify(batchJob));
  
  // Prepare inputs for processing
  const preparedInputs = await prepareInputs(driver, batchInputs);
  
  // Start the batch processing in the background
  processBatchJob(driver, jobId, preparedInputs, concurrency, batchSize, retryCount, retryDelay, timeout)
    .catch(error => {
      driver.logger.error(`Error processing batch job ${jobId}: ${error}`);
      
      // Update job status to failed
      const updatedJob = getBatchInferenceJob(driver, jobId);
      if (updatedJob) {
        updatedJob.status = BatchInferenceJobStatus.failed;
        updatedJob.details = `Error: ${error.message}`;
        updatedJob.end_time = new Date();
        
        fs.writeFileSync(batchJobFile, JSON.stringify(updatedJob));
      }
    });
  
  return batchJob;
}

/**
 * Process the batch job asynchronously
 */
async function processBatchJob(
  driver: BaseOpenAIDriver,
  jobId: string,
  preparedInputs: { id: string, options: ExecutionOptions, prompt: any }[],
  concurrency: number,
  batchSize: number,
  retryCount: number,
  retryDelay: number,
  timeout: number
): Promise<void> {
  const batchDir = path.join(os.tmpdir(), 'llumiverse-openai-batch', jobId);
  const batchJobFile = path.join(batchDir, 'job.json');
  const batchResultsFile = path.join(batchDir, 'results.json');
  
  // Update job status to running
  let batchJob = await getBatchInferenceJob(driver, jobId);
  if (!batchJob) {
    throw new Error(`Batch job ${jobId} not found`);
  }
  
  batchJob.status = BatchInferenceJobStatus.running;
  fs.writeFileSync(batchJobFile, JSON.stringify(batchJob));
  
  // Initialize results
  const batchResults: BatchItemResult[] = [];
  
  try {
    // Split inputs into chunks of the specified batch size
    const chunks = [];
    for (let i = 0; i < preparedInputs.length; i += batchSize) {
      chunks.push(preparedInputs.slice(i, i + batchSize));
    }
    
    // Process chunks with limited concurrency
    let completedCount = 0;
    let errorCount = 0;
    
    for (let i = 0; i < chunks.length; i += concurrency) {
      const concurrentChunks = chunks.slice(i, i + concurrency);
      
      // Process chunks concurrently
      const chunkResults = await Promise.all(
        concurrentChunks.map(chunk => 
          processChunk(driver, chunk, retryCount, retryDelay, timeout)
        )
      );
      
      // Flatten and add results
      for (const results of chunkResults) {
        for (const result of results) {
          batchResults.push(result);
          if (result.error) {
            errorCount++;
          } else {
            completedCount++;
          }
        }
      }
      
      // Update job status with progress
      batchJob.completed_count = completedCount;
      batchJob.error_count = errorCount;
      fs.writeFileSync(batchJobFile, JSON.stringify(batchJob));
      
      // Save partial results
      fs.writeFileSync(batchResultsFile, JSON.stringify(batchResults));
    }
    
    // Update job status to succeeded or partial
    batchJob.status = errorCount > 0 ? BatchInferenceJobStatus.partial : BatchInferenceJobStatus.succeeded;
    batchJob.end_time = new Date();
    batchJob.details = errorCount > 0 
      ? `Completed with ${errorCount} errors out of ${preparedInputs.length} requests` 
      : `Successfully processed all ${preparedInputs.length} requests`;
    
    fs.writeFileSync(batchJobFile, JSON.stringify(batchJob));
    fs.writeFileSync(batchResultsFile, JSON.stringify(batchResults));
  } catch (error) {
    // Handle unexpected errors
    batchJob.status = BatchInferenceJobStatus.failed;
    batchJob.end_time = new Date();
    batchJob.details = `Error: ${error instanceof Error ? error.message : String(error)}`;
    
    fs.writeFileSync(batchJobFile, JSON.stringify(batchJob));
    
    throw error;
  }
}

/**
 * Get the status of a batch inference job
 */
export function getBatchInferenceJob(
  driver: BaseOpenAIDriver,
  jobId: string
): BatchInferenceJob {
  try {
    const batchDir = path.join(os.tmpdir(), 'llumiverse-openai-batch', jobId);
    const batchJobFile = path.join(batchDir, 'job.json');
    
    if (!fs.existsSync(batchJobFile)) {
      throw new Error(`Batch job ${jobId} not found`);
    }
    
    const jobData = fs.readFileSync(batchJobFile, 'utf8');
    const batchJob = JSON.parse(jobData) as BatchInferenceJob;
    
    // Convert date strings back to Date objects
    if (batchJob.start_time) {
      batchJob.start_time = new Date(batchJob.start_time);
    }
    if (batchJob.end_time) {
      batchJob.end_time = new Date(batchJob.end_time);
    }
    
    return batchJob;
  } catch (error) {
    driver.logger.error(`Error getting batch job ${jobId}: ${error}`);
    throw error;
  }
}

/**
 * Cancel a running batch inference job
 * Note: Since OpenAI batch jobs run locally in the process, cancelling
 * just marks the job as cancelled but doesn't actually stop processing
 */
export function cancelBatchInferenceJob(
  driver: BaseOpenAIDriver,
  jobId: string
): BatchInferenceJob {
  try {
    const batchJob = getBatchInferenceJob(driver, jobId);
    
    if (batchJob.status === BatchInferenceJobStatus.running ||
        batchJob.status === BatchInferenceJobStatus.created) {
      batchJob.status = BatchInferenceJobStatus.cancelled;
      batchJob.end_time = new Date();
      batchJob.details = `Job cancelled at ${batchJob.end_time.toISOString()}`;
      
      const batchDir = path.join(os.tmpdir(), 'llumiverse-openai-batch', jobId);
      const batchJobFile = path.join(batchDir, 'job.json');
      fs.writeFileSync(batchJobFile, JSON.stringify(batchJob));
    }
    
    return batchJob;
  } catch (error) {
    driver.logger.error(`Error cancelling batch job ${jobId}: ${error}`);
    throw error;
  }
}

/**
 * Get the results of a batch inference job
 */
export function getBatchInferenceResults(
  driver: BaseOpenAIDriver,
  jobId: string,
  options?: { maxResults?: number }
): BatchInferenceResult {
  try {
    const batchJob = getBatchInferenceJob(driver, jobId);
    const batchDir = path.join(os.tmpdir(), 'llumiverse-openai-batch', jobId);
    const batchResultsFile = path.join(batchDir, 'results.json');
    
    const completed = batchJob.status === BatchInferenceJobStatus.succeeded || 
                      batchJob.status === BatchInferenceJobStatus.failed ||
                      batchJob.status === BatchInferenceJobStatus.cancelled ||
                      batchJob.status === BatchInferenceJobStatus.partial;
    
    if (!completed) {
      return {
        batch_id: jobId,
        results: [],
        completed: false,
        error: 'Batch job is still running'
      };
    }
    
    if (batchJob.status === BatchInferenceJobStatus.failed) {
      return {
        batch_id: jobId,
        results: [],
        completed: true,
        error: batchJob.details || 'Batch job failed'
      };
    }
    
    // Read results file
    if (!fs.existsSync(batchResultsFile)) {
      return {
        batch_id: jobId,
        results: [],
        completed: true,
        error: 'Results file not found'
      };
    }
    
    const resultsData = fs.readFileSync(batchResultsFile, 'utf8');
    let results = JSON.parse(resultsData) as BatchItemResult[];
    
    // Apply maxResults limit if provided
    if (options?.maxResults && options.maxResults > 0) {
      results = results.slice(0, options.maxResults);
    }
    
    return {
      batch_id: jobId,
      results,
      completed: true,
      error: batchJob.status === BatchInferenceJobStatus.partial ? 'Some items failed processing' : undefined
    };
  } catch (error) {
    driver.logger.error(`Error getting batch results for job ${jobId}: ${error}`);
    throw error;
  }
}