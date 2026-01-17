using Microsoft.CodeAnalysis.CSharp;

var path = "path/to/csharp_files";
var outputCsv = "path/to/output/roslyn_analysis.csv";

var files = Directory.GetFiles(path, "*.cs");
Console.WriteLine($"Found {files.Length} C# files\n");

var results = new List<(string TaskId, int SyntaxErrors, int Maintainability)>();

int totalSyntaxErrors = 0;
int totalMaintainability = 0;

foreach (var file in files)
{
    var code = File.ReadAllText(file);
    var tree = CSharpSyntaxTree.ParseText(code);
    
    int syntaxErrors = 0;
    int maintainabilityIssues = 0;
    
    foreach (var error in tree.GetDiagnostics())
    {
        if (error.Id.StartsWith("CS01")) 
            syntaxErrors++;
        else 
            maintainabilityIssues++;
    }
    
    // Extract task ID from filename
    var taskId = Path.GetFileNameWithoutExtension(file);
    
    results.Add((taskId, syntaxErrors, maintainabilityIssues));
    
    totalSyntaxErrors += syntaxErrors;
    totalMaintainability += maintainabilityIssues;
}

// Sort by Task ID
results = results
    .OrderBy(r => {
        if (r.TaskId.StartsWith("translation_"))
        {
            var numPart = r.TaskId.Replace("translation_", "");
            if (int.TryParse(numPart, out int num))
                return num;
        }
        return 0;
    })
    .ThenBy(r => r.TaskId)
    .ToList();

using (var writer = new StreamWriter(outputCsv))
{
    writer.WriteLine("Task ID,Syntax_Errors,Maintainability");
    foreach (var r in results)
    {
        writer.WriteLine($"{r.TaskId},{r.SyntaxErrors},{r.Maintainability}");
    }
}

Console.WriteLine($"CSV saved to: {outputCsv}");
Console.WriteLine($"\nTotal files: {results.Count}");
Console.WriteLine($"Sum Syntax Errors: {totalSyntaxErrors}");
Console.WriteLine($"Sum Maintainability: {totalMaintainability}");
