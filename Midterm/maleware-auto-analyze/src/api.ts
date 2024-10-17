import fs from "fs";
import path from "path";
import axios from "axios";
import FormData from "form-data";

export interface Task {
  category: string; // "url"
  machine: any | null;
  errors: string[];
  target: any | null;
  package: any | null;
  sample_id: any | null;
  guest: Record<string, any>; // Empty object, can define more specific type if known
  custom: any | null;
  owner: string;
  priority: number;
  platform: any | null;
  options: any | null;
  status: string; // "pending"
  enforce_timeout: boolean;
  timeout: number;
  memory: boolean;
  tags: string[];
  id: number;
  added_on: string; // "2012-12-19 14:18:25"
  completed_on: any | null;
}

// 自定義錯誤類別
class FileUploadError extends Error {}
class TaskStatusError extends Error {}
class ReportDownloadError extends Error {}
class PcapDownloadError extends Error {}

export class CuckooClient {
  private apiBaseUrl: string;
  private token: string;

  constructor(token: string, apiBaseUrl: string = "http://localhost:8080/") {
    this.apiBaseUrl = apiBaseUrl;
    this.token = token;
  }

  private getHeaders() {
    return {
      Authorization: `Bearer ${this.token}`,
    };
  }

  // 上傳並分析單個檔案
  public async uploadAndAnalyzeFile(filePath: string): Promise<number> {
    try {
      const fileName = path.basename(filePath);
      const formData = new FormData();
      formData.append("file", fs.createReadStream(filePath), fileName);

      const url = new URL("tasks/create/file", this.apiBaseUrl);

      const response = await axios.post(url.toString(), formData, {
        headers: this.getHeaders(),
      });
      if (response.status !== 200) {
        throw new FileUploadError(
          `Failed to upload ${filePath}. Status code: ${response.status}`
        );
      }

      const taskId = response.data.task_id;
      if (!taskId) {
        throw new FileUploadError(
          `Failed to retrieve task_id for ${filePath}.`
        );
      }
      return taskId;
    } catch (error) {
      if (error instanceof Error) {
        throw new FileUploadError(
          `Error uploading file ${filePath}: ${error.message}`
        );
      } else {
        throw error;
      }
    }
  }

  // 查詢任務狀態
  public async checkTaskStatus(
    taskId: number
  ): Promise<
    "pending" | "running" | "completed" | "reported" | "failed" | string
  > {
    try {
      const url = new URL(`/tasks/view/${taskId}`, this.apiBaseUrl);

      const response = await axios.get(url.toString(), {
        headers: this.getHeaders(),
      });
      if (response.status !== 200) {
        throw new TaskStatusError(
          `Failed to get task status for ${taskId}. Status code: ${response.status}`
        );
      }
      return response.data.task.status;
    } catch (error) {
      if (error instanceof Error) {
        throw new TaskStatusError(
          `Error checking task status for ${taskId}: ${error.message}`
        );
      } else {
        throw error;
      }
    }
  }

  // 下載分析報告和PCAP
  public async downloadReportAndPcap(
    taskId: number,
    outputFolder: string
  ): Promise<void> {
    const reportUrl = new URL(`/tasks/report/${taskId}/json`, this.apiBaseUrl);
    const pcapUrl = new URL(`/pcap/get/${taskId}`, this.apiBaseUrl);

    try {
      const reportResponse = await axios.get(reportUrl.toString(), {
        headers: this.getHeaders(),
      });
      if (reportResponse.status !== 200) {
        throw new ReportDownloadError(
          `Failed to download report for task ${taskId}. Status code: ${reportResponse.status}`
        );
      }
      const reportPath = path.join(outputFolder, `report_${taskId}.json`);
      fs.writeFileSync(
        reportPath,
        JSON.stringify(reportResponse.data, null, 2)
      );
    } catch (error) {
      if (error instanceof Error) {
        throw new ReportDownloadError(
          `Error downloading report for task ${taskId}: ${error.message}`
        );
      } else {
        throw error;
      }
    }

    try {
      const pcapResponse = await axios.get(pcapUrl.toString(), {
        headers: this.getHeaders(),
        responseType: "stream",
      });
      if (pcapResponse.status !== 200) {
        throw new PcapDownloadError(
          `Failed to download PCAP for task ${taskId}. Status code: ${pcapResponse.status}`
        );
      }
      const pcapPath = path.join(outputFolder, `dump_sorted_${taskId}.pcap`);
      const pcapStream = fs.createWriteStream(pcapPath);
      pcapResponse.data.pipe(pcapStream);
    } catch (error) {
      if (error instanceof Error) {
        throw new PcapDownloadError(
          `Error downloading PCAP for task ${taskId}: ${error.message}`
        );
      } else {
        throw error;
      }
    }
  }

  public async getTaskList(): Promise<Task[]> {
    try {
      const url = new URL("/tasks/list", this.apiBaseUrl);

      const response = await axios.get(url.toString(), {
        headers: this.getHeaders(),
      });
      if (response.status !== 200) {
        throw new TaskStatusError(
          `Failed to get task list. Status code: ${response.status}`
        );
      }
      return response.data.tasks;
    } catch (error) {
      if (error instanceof Error) {
        throw new TaskStatusError(`Error getting task list: ${error.message}`);
      } else {
        throw error;
      }
    }
  }
}
