import fs from "fs/promises";
import { program } from "commander";
import path from "path";
import { getMatchedNames, type TaskInfo } from "./common";
import { CuckooClient } from "./api";
import { sleep } from "bun";

let inputDir: string | null = null;
let outputDir: string | null = null;
let labelListPath: string | null = null;
let targetLabel: string | null = null;
let skipExists = true;

program
  .version("1.0.0")
  .description("maleware-auto-analyze")
  .requiredOption("-i, --input-dir <inputDir>", "Input malewares directory")
  .requiredOption(
    "-l, --label-list <labelList>",
    "Malewares label list csv file"
  )
  .requiredOption("-t, --target-label <targetlabel>", "Target maleware label")
  .requiredOption("-o, --output-dir <outputDir>", "Output directory")
  .option("--skip-exists", "Skip the malewares that already analyzed", true)
  .action((options) => {
    inputDir = options.inputDir;
    outputDir = options.outputDir;
    labelListPath = options.labelList;
    targetLabel = options.targetLabel;
    skipExists = options.skipExists;
  });
program.parse(process.argv);

if (!inputDir || !outputDir || !labelListPath || !targetLabel) {
  console.error("Error: Missing required options");
  process.exit(1);
}

console.log("Matching malewares with label...");

let malewareBinaries: { name: string; path: string }[] = [];

// Filter the maleware by label
const labelList = await getMatchedNames(labelListPath, targetLabel);

// Get all files path that belong to the label
for (const filename of await fs.readdir(inputDir)) {
  if (!(await fs.stat(path.join(inputDir, filename))).isFile()) {
    // Skip directories
    continue;
  }

  if (labelList.includes(filename)) {
    malewareBinaries.push({
      name: filename,
      path: path.join(inputDir, filename),
    });
  }
}

if (skipExists) {
  // Filter the maleware that already analyzed
  const analyzedTasks = await fs.readdir(outputDir);
  malewareBinaries = malewareBinaries.filter(
    (maleware) => !analyzedTasks.includes(maleware.name)
  );
  console.log(
    `Found ${malewareBinaries.length} malewares labeled ${targetLabel} in ${inputDir}, skiped ${analyzedTasks.length} malewares analyzed`
  );
} else {
  console.log(
    `Found ${malewareBinaries.length} malewares labeled ${targetLabel} in ${inputDir}`
  );
}

const API_BASE_URL = "http://192.168.2.128:8090/";
const API_TOKEN = "UdC5HKZB1aruy8e-Giv_fg";

const client = new CuckooClient(API_TOKEN, API_BASE_URL);

const runningTasks: TaskInfo[] = [];
const tasks: TaskInfo[] = [];

// Upload all files
for (const maleware of malewareBinaries) {
  console.log(`Uploading ${maleware.name}...`);
  const taskId = await client.uploadAndAnalyzeFile(maleware.path);
  console.log(`Task ${taskId} created for ${maleware.name}`);
  tasks.push({ taskId, name: maleware.name });
}

// Add all tasks to runningTasks
runningTasks.push(...tasks);

while (runningTasks.length > 0) {
  // Extract the tasks that belong to me
  const myTasks = (await client.getTaskList()).filter((task) =>
    tasks.some((t) => t.taskId === task.id)
  );

  // Extract the tasks that the report not downloaded
  const myUnprocessedTasks = myTasks.filter((task) =>
    runningTasks.some((t) => t.taskId === task.id)
  );

  // Extract the tasks that the report generated
  const myFinishTasks = myUnprocessedTasks.filter(
    (task) => task.status === "reported"
  );

  // Download the report and pcap for each task
  for (const task of myFinishTasks) {
    const name = tasks.find((t) => t.taskId === task.id)?.name;
    if (!name) {
      console.error(`Failed to find name for task ${task.id}`);
      continue;
    }

    const outputFolder = path.join(outputDir, name);

    // Create the output folder if it doesn't exist
    if (!(await fs.exists(outputFolder))) {
      await fs.mkdir(outputFolder, { recursive: true });
    }

    // Download the report and pcap
    await client.downloadReportAndPcap(task.id, outputFolder);
    console.log(`Report for task ${task.id} downloaded to ${outputFolder}`);

    // Remove the task from runningTasks
    runningTasks.splice(
      runningTasks.findIndex((t) => t.taskId === task.id),
      1
    );
  }

  // Get the status of all tasks
  const myPendingTasks = myTasks.filter((task) => task.status === "pending");
  const myRunningTasks = myTasks.filter((task) => task.status === "running");
  const myCompletedTasks = myTasks.filter(
    (task) => task.status === "completed"
  );
  const myReportedTasks = myTasks.filter((task) => task.status === "reported");

  const completedRate = Math.round(
    (myReportedTasks.length / myTasks.length) * 100
  );

  console.log(
    `Total: ${tasks.length} | Pending: ${myPendingTasks.length} | Running: ${myRunningTasks.length} | Completed: ${myCompletedTasks.length} | Reported: ${myReportedTasks.length} | Completed Rate: ${completedRate}%`
  );

  await sleep(1000);
}
