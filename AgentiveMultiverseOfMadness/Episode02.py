import os
import json
import time
from agentarium import Agent, AgentInteractionManager

class Task:
    def __init__(self, data):
        self.data = data
class SourceCodeAnalysisAgent:
    def perform(self, task):
        directory = task['directory']
        report_path = task['report_path']
        analysis = {
            "code_structure": self.analyze_code_structure(directory),
            "dependencies": self.map_dependencies(directory),
            "performance": self.analyze_performance(directory),
            "business_logic": self.extract_business_logic(directory),
        }
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=4)
        return {"status": "success", "report_path": report_path}


    def analyze_code_structure(self, directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(('.cpp', '.h', '.hpp', '.c', '.cc')):
                    files.append(os.path.join(root, filename))

        if not files:
            return {"error": "No C++ source files found in the directory."}

        prompt = (
            "Analyze the following C++ files for their structure, including hierarchical relationships, "
            "inheritance, and design patterns. Provide the results in a structured JSON format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=1500
            )

            analysis = json.loads(response['choices'][0]['text'].strip())
            return analysis
        except Exception as e:
            print(f"Error during LLM code structure analysis: {e}")
            return {"error": str(e)}


    def map_dependencies(self, directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(('.cpp', '.h', '.hpp', '.c', '.cc')):
                    files.append(os.path.join(root, filename))

        if not files:
            return {"error": "No C++ source files found in the directory."}

        prompt = (
            "Map the dependencies in the following C++ files. Identify external libraries, inter-class and inter-module "
            "relationships, potential bottlenecks, and tight couplings. Provide the results in a structured JSON format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=1500
            )

            dependencies = json.loads(response['choices'][0]['text'].strip())
            return dependencies
        except Exception as e:
            print(f"Error during LLM dependency mapping: {e}")
            return {"error": str(e)}


    def analyze_performance(self, directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(('.cpp', '.h', '.hpp', '.c', '.cc')):
                    files.append(os.path.join(root, filename))

        if not files:
            return {"error": "No C++ source files found in the directory."}

        prompt = (
            "Analyze the performance characteristics of the following C++ files. Identify memory management strategies, "
            "threading models, resource allocation and deallocation, and algorithm complexities. Provide the results "
            "in a structured JSON format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=1500
            )

            performance = json.loads(response['choices'][0]['text'].strip())
            return performance
        except Exception as e:
            print(f"Error during LLM performance analysis: {e}")
            return {"error": str(e)}


    def extract_business_logic(self, directory):
        """
        Extracts business logic from the codebase using an LLM.

        Parameters:
        - directory (str): Path to the codebase directory.

        Returns:
        - dict: Details about core business rules, critical processes, and configuration strategies.
        """
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(('.cpp', '.h', '.hpp', '.c', '.cc')):
                    files.append(os.path.join(root, filename))

        if not files:
            return {"error": "No C++ source files found in the directory."}

        prompt = (
            "Extract the business logic from the following C++ files. Identify core business rules, critical processes, "
            "and configuration strategies. Provide the results in a structured JSON format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=1500
            )

            business_logic = json.loads(response['choices'][0]['text'].strip())
            return business_logic
        except Exception as e:
            print(f"Error during LLM business logic extraction: {e}")
            return {"error": str(e)}

class SourceCodeMigrationAgent:
    def perform(self, task):
        input_directory = task['input_directory']
        output_directory = task['output_directory']
        self.translate_code(input_directory, output_directory)
        self.adapt_to_spring_boot(output_directory)
        self.optimize_performance(output_directory)
        return {"status": "success", "output_directory": output_directory}


    def translate_code(self, input_directory, output_directory):
        """
        Translates C++ code to Java using an LLM.

        Parameters:
        - input_directory (str): Path to the directory containing C++ code.
        - output_directory (str): Path to the directory where translated Java code will be saved.

        Returns:
        - None
        """
        files = []
        for root, _, filenames in os.walk(input_directory):
            for filename in filenames:
                if filename.endswith(('.cpp', '.h', '.hpp', '.c', '.cc')):
                    files.append(os.path.join(root, filename))

        if not files:
            print("No C++ source files found for translation.")
            return

        prompt = (
            "Translate the following C++ files to Java. Ensure proper handling of memory management, template conversion, "
            "and other language-specific features. Provide the translated code in Java format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=3000
            )

            translations = json.loads(response['choices'][0]['text'].strip())
            os.makedirs(output_directory, exist_ok=True)
            for filename, code in translations.items():
                with open(os.path.join(output_directory, filename), 'w') as f:
                    f.write(code)
        except Exception as e:
            print(f"Error during code translation: {e}")


    def adapt_to_spring_boot(self, output_directory):
        """
        Adapts Java code to Spring Boot standards using an LLM.

        Parameters:
        - output_directory (str): Path to the directory containing translated Java code.

        Returns:
        - None
        """
        files = []
        for root, _, filenames in os.walk(output_directory):
            for filename in filenames:
                if filename.endswith('.java'):
                    files.append(os.path.join(root, filename))

        if not files:
            print("No Java files found for Spring Boot adaptation.")
            return

        prompt = (
            "Adapt the following Java code to Spring Boot standards. Include appropriate annotations and dependency "
            "injection patterns. Provide the adapted code in Java format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=3000
            )

            adaptations = json.loads(response['choices'][0]['text'].strip())
            for filename, code in adaptations.items():
                with open(os.path.join(output_directory, filename), 'w') as f:
                    f.write(code)
        except Exception as e:
            print(f"Error during Spring Boot adaptation: {e}")


    def optimize_performance(self, output_directory):
        """
        Optimizes the performance of the translated Java code using an LLM.

        Parameters:
        - output_directory (str): Path to the directory containing Java code.

        Returns:
        - None
        """
        files = []
        for root, _, filenames in os.walk(output_directory):
            for filename in filenames:
                if filename.endswith('.java'):
                    files.append(os.path.join(root, filename))

        if not files:
            print("No Java files found for performance optimization.")
            return

        prompt = (
            "Optimize the performance of the following Java files. Focus on improving memory usage, concurrency handling, "
            "and computational efficiency. Provide the optimized code in Java format:\n\n"
            + "\n".join(files)
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=3000
            )

            optimizations = json.loads(response['choices'][0]['text'].strip())
            for filename, code in optimizations.items():
                with open(os.path.join(output_directory, filename), 'w') as f:
                    f.write(code)
        except Exception as e:
            print(f"Error during performance optimization: {e}")


# Migration Validation Agent
class MigrationValidationAgent:
    def perform(self, task):
        input_directory = task['input_directory']
        output_directory = task['output_directory']
        report_path = task['validation_report_path']

        validation_report = {
            "functional_verification": self.verify_functionality(input_directory, output_directory),
            "performance_benchmarking": self.benchmark_performance(input_directory, output_directory),
            "migration_integrity": self.check_migration_integrity(input_directory, output_directory),
            "non_functional_testing": self.test_non_functional_requirements(output_directory),
        }

        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=4)

        return {"status": "success", "report_path": report_path}


    def verify_functionality(self, input_directory, output_directory):
        """
        Verifies the functional equivalence between the original and migrated codebases using an LLM.

        Parameters:
        - input_directory (str): Path to the original C++ codebase.
        - output_directory (str): Path to the migrated Java codebase.

        Returns:
        - dict: Details of functional equivalence test results.
        """
        prompt = (
            "Verify the functional equivalence between the following original C++ codebase and the migrated Java codebase. "
            "Highlight any discrepancies in functionality and provide a structured report:\n\n"
            f"Original Codebase: {input_directory}\nMigrated Codebase: {output_directory}\n"
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=2000
            )

            return json.loads(response['choices'][0]['text'].strip())
        except Exception as e:
            print(f"Error during functionality verification: {e}")
            return {"error": str(e)}


    def benchmark_performance(self, input_directory, output_directory):
        """
        Benchmarks the performance of the migrated code against the original codebase using an LLM.

        Parameters:
        - input_directory (str): Path to the original C++ codebase.
        - output_directory (str): Path to the migrated Java codebase.

        Returns:
        - dict: Performance comparison results.
        """
        prompt = (
            "Benchmark the performance of the following original C++ codebase against the migrated Java codebase. "
            "Provide a detailed comparison of execution times, memory usage, and scalability:\n\n"
            f"Original Codebase: {input_directory}\nMigrated Codebase: {output_directory}\n"
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=2000
            )

            return json.loads(response['choices'][0]['text'].strip())
        except Exception as e:
            print(f"Error during performance benchmarking: {e}")
            return {"error": str(e)}


    def check_migration_integrity(self, input_directory, output_directory):
        """
        Checks the integrity of the migration process to ensure no logic was altered during translation.

        Parameters:
        - input_directory (str): Path to the original C++ codebase.
        - output_directory (str): Path to the migrated Java codebase.

        Returns:
        - dict: Migration integrity validation results.
        """
        prompt = (
            "Check the migration integrity between the following original C++ codebase and the migrated Java codebase. "
            "Ensure that all logic and functionality remain consistent post-migration:\n\n"
            f"Original Codebase: {input_directory}\nMigrated Codebase: {output_directory}\n"
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=2000
            )

            return json.loads(response['choices'][0]['text'].strip())
        except Exception as e:
            print(f"Error during migration integrity check: {e}")
            return {"error": str(e)}


    def test_non_functional_requirements(self, output_directory):
        """
        Tests non-functional requirements such as scalability, concurrency handling, and error management.

        Parameters:
        - output_directory (str): Path to the migrated Java codebase.

        Returns:
        - dict: Results of non-functional requirement testing.
        """
        prompt = (
            "Test the non-functional requirements of the following migrated Java codebase. "
            "Focus on scalability, concurrency handling, memory utilization, and error management. "
            "Provide a structured report:\n\n"
            f"Migrated Codebase: {output_directory}\n"
        )

        try:
            response = Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=2000
            )

            return json.loads(response['choices'][0]['text'].strip())
        except Exception as e:
            print(f"Error during non-functional requirement testing: {e}")
            return {"error": str(e)}


class CodeMigrationManagementAgent(Agent):
    def perform(self, task: Task):
        """
        Orchestrates and oversees the migration process by coordinating with multiple agents.

        Parameters:
        - task (Task): Contains:
            - tasks (dict): A mapping of agent names to their respective task data.
            - report_path (str): Path to save the migration management report.

        Returns:
        - dict: Contains the status and the path to the generated report.
        """
        tasks = task.data['tasks']
        report_path = task.data['report_path']

        management_report = {
            "task_timings": {},
            "improvement_suggestions": []
        }

        for agent_name, agent_task in tasks.items():
            start_time = time.time()
            result = self.system.run_task(agent_name, Task(data=agent_task))
            end_time = time.time()

            management_report["task_timings"][agent_name] = end_time - start_time

            if result.get("status") != "success":
                management_report["improvement_suggestions"].append(
                    {
                        "agent": agent_name,
                        "issue": result.get("message", "Unknown error"),
                        "suggested_fix": "Check task inputs and dependencies."
                    }
                )

        self.generate_report(management_report, report_path)
        return {"status": "success", "report_path": report_path}

    def generate_report(self, report: dict, path: str):
        """
        Generates a detailed migration management report.

        Parameters:
        - report (dict): The report data to be written to a file.
        - path (str): The file path where the report will be saved.

        Returns:
        - None
        """
        try:
            with open(path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Management report saved to {path}")
        except Exception as e:
            print(f"Error generating report: {e}")


if __name__ == "__main__":
    input_directory = "/path/to/old_code"
    output_directory = "/path/to/migrated_code"
    report_directory = "/path/to/reports"
    os.makedirs(report_directory, exist_ok=True)

    system = CodeMigrationSystem()

    task_data = {
        "tasks": {
            "code_analysis": {"directory": input_directory, "report_path": os.path.join(report_directory, "analysis_report.json")},
            "code_migration": {"input_directory": input_directory, "output_directory": output_directory},
            "code_validation": {"input_directory": input_directory, "output_directory": output_directory, "validation_report_path": os.path.join(report_directory, "validation_report.json")},
        },
        "report_path": os.path.join(report_directory, "migration_management_report.txt"),
    }

    result = system.run_task("management", Task(data=task_data))
    if result["status"] == "success":
        print(f"Migration completed. Report generated at {result['report_path']}")
    else:
        print(f"Migration failed: {result.get('message', 'Unknown error')}")