import { spawn } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const BIN_PATHS = {
  next: require.resolve("next/dist/bin/next"),
  tsc: require.resolve("typescript/bin/tsc"),
};

const [, , commandName, ...commandArgs] = process.argv;

if (!commandName || !(commandName in BIN_PATHS)) {
  console.error(`Unsupported command "${commandName ?? ""}". Expected one of: ${Object.keys(BIN_PATHS).join(", ")}.`);
  process.exit(1);
}

function isDebuggerBootloaderPath(value) {
  return /js-debug|ms-vscode\.js-debug|bootloader/i.test(value);
}

function sanitizeNodeOptions(rawValue) {
  if (!rawValue) {
    return undefined;
  }

  const tokens = rawValue.match(/(?:[^\s"]+|"[^"]*")+/g) ?? [];
  const sanitizedTokens = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    const normalizedToken = token.replace(/^"|"$/g, "");

    if (/^--inspect(?:-brk|-port)?(?:=.*)?$/.test(normalizedToken)) {
      continue;
    }

    if (normalizedToken === "--inspect-publish-uid") {
      index += 1;
      continue;
    }

    if (normalizedToken.startsWith("--inspect-publish-uid=")) {
      continue;
    }

    if (normalizedToken === "--require") {
      const nextToken = tokens[index + 1]?.replace(/^"|"$/g, "");

      if (nextToken && isDebuggerBootloaderPath(nextToken)) {
        index += 1;
        continue;
      }

      sanitizedTokens.push(token);
      continue;
    }

    if (normalizedToken.startsWith("--require=") && isDebuggerBootloaderPath(normalizedToken.slice("--require=".length))) {
      continue;
    }

    sanitizedTokens.push(token);
  }

  return sanitizedTokens.length > 0 ? sanitizedTokens.join(" ") : undefined;
}

const childEnv = { ...process.env };
const sanitizedNodeOptions = sanitizeNodeOptions(childEnv.NODE_OPTIONS);

if (sanitizedNodeOptions) {
  childEnv.NODE_OPTIONS = sanitizedNodeOptions;
} else {
  delete childEnv.NODE_OPTIONS;
}

delete childEnv.VSCODE_INSPECTOR_OPTIONS;
delete childEnv.NODE_INSPECT_RESUME_ON_START;

const childProcess = spawn(process.execPath, [BIN_PATHS[commandName], ...commandArgs], {
  stdio: "inherit",
  env: childEnv,
});

childProcess.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 0);
});
