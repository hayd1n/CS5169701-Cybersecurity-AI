import csvParser from "csv-parser";
import fs from "fs";

export interface TaskInfo {
  name: string;
  taskId: number;
}

export async function getMatchedNames(
  csvFilePath: string,
  label: string
): Promise<string[]> {
  const matchedNames: string[] = [];

  return new Promise((resolve, reject) => {
    const stream = fs
      .createReadStream(csvFilePath)
      .pipe(csvParser())
      .on("data", (row) => {
        if (row.label === label) {
          matchedNames.push(row.name);
        }
      })
      .on("end", () => {
        resolve(matchedNames);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}
