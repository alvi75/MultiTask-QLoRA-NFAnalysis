import os
import subprocess
import json
import pandas as pd
import re
from collections import defaultdict

def extract_checkstyle_issues(report_path, category_to_rules):
    """Extract and categorize Checkstyle issues from a report file."""
    categorized_issues = defaultdict(list)
    error_code_counts = defaultdict(lambda: defaultdict(int))
    
    with open(report_path, 'r') as file:
        for line in file:
            match = re.search(r"\[(\w+)]$", line)
            if match:
                rule_name = match.group(1)
                issue_detail = line.split(":")[-2].strip() + ": " + line.split(":")[-1].strip()
                for category, rules in category_to_rules.items():
                    if rule_name in rules:
                        categorized_issues[category].append(f"{issue_detail} [{rule_name}]")
                        error_code_counts[category][rule_name] += 1
                        break
    return categorized_issues, error_code_counts

def extract_pmd_violations(report_path):
    """Extract PMD violations from a report file."""
    violations = []
    violation_counts = defaultdict(int)
    
    # Check if file exists and has content
    if not os.path.exists(report_path):
        return violations, violation_counts
        
    with open(report_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # PMD output format: filepath:line:column:\t<message>
            parts = line.split(":\t")
            if len(parts) > 1:
                category_message = parts[1].strip()
                violations.append(category_message)
                violation_counts[category_message] += 1
    return violations, violation_counts

def run_checkstyle_analysis(input_folder, checkstyle_output_dir):
    os.makedirs(checkstyle_output_dir, exist_ok=True)
    results = []
    checkstyle_issues = defaultdict(list)
    error_code_counts = defaultdict(lambda: defaultdict(int))
    checkstyle_jar = os.path.expanduser("~/checkstyle/checkstyle-10.21.1-all.jar")
    config_file = os.path.expanduser("~/checkstyle/sun_checks.xml")
    
    category_to_rules = {
        "annotation": ["AnnotationLocation", "AnnotationOnSameLine", "AnnotationUseStyle", "MissingDeprecated", "MissingOverride", "PackageAnnotation"],
        "blocks": ["AvoidNestedBlocks", "EmptyBlock", "EmptyCatchBlock", "LeftCurly", "NeedBraces", "RightCurly"],
        "coding": ["ArrayTrailingComma", "AvoidDoubleBraceInitialization", "AvoidInlineConditionals", "AvoidNoArgumentSuperConstructorCall", "ConstructorsDeclarationGrouping", "CovariantEquals", "DeclarationOrder", "DefaultComesLast", "EmptyStatement", "EqualsAvoidNull", "EqualsHashCode", "ExplicitInitialization", "FallThrough", "FinalLocalVariable", "HiddenField", "IllegalCatch", "IllegalInstantiation", "IllegalThrows", "IllegalToken", "IllegalTokenText", "IllegalType", "InnerAssignment", "MagicNumber", "MatchXpath", "MissingCtor", "MissingNullCaseInSwitch", "MissingSwitchDefault", "ModifiedControlVariable", "MultipleStringLiterals", "MultipleVariableDeclarations", "NestedForDepth", "NestedIfDepth", "NestedTryDepth", "NoArrayTrailingComma", "NoClone", "NoEnumTrailingComma", "NoFinalizer", "OneStatementPerLine", "OverloadMethodsDeclarationOrder", "PackageDeclaration", "ParameterAssignment", "RequireThis", "ReturnCount", "SimplifyBooleanExpression", "SimplifyBooleanReturn", "StringLiteralEquality", "SuperClone", "SuperFinalize", "UnnecessaryParentheses", "UnnecessarySemicolonAfterOuterTypeDeclaration", "UnnecessarySemicolonAfterTypeMemberDeclaration", "UnnecessarySemicolonInEnumeration", "UnnecessarySemicolonInTryWithResources", "UnusedCatchParameterShouldBeUnnamed", "UnusedLambdaParameterShouldBeUnnamed", "UnusedLocalVariable", "VariableDeclarationUsageDistance", "WhenShouldBeUsed"],
        "design": ["DesignForExtension", "FinalClass", "HideUtilityClassConstructor", "InnerTypeLast", "InterfaceIsType", "MutableException", "OneTopLevelClass", "SealedShouldHavePermitsList", "ThrowsCount", "VisibilityModifier"],
        "imports": ["AvoidStarImport", "AvoidStaticImport", "CustomImportOrder", "IllegalImport", "ImportControl", "ImportOrder", "RedundantImport", "UnusedImports"],
        "javadoc": ["AtclauseOrder", "InvalidJavadocPosition", "JavadocBlockTagLocation", "JavadocContentLocation", "JavadocLeadingAsteriskAlign", "JavadocMethod", "JavadocMissingLeadingAsterisk", "JavadocMissingWhitespaceAfterAsterisk", "JavadocPackage", "JavadocParagraph", "JavadocStyle", "JavadocTagContinuationIndentation", "JavadocType", "JavadocVariable", "MissingJavadocMethod", "MissingJavadocPackage", "MissingJavadocType", "NonEmptyAtclauseDescription", "RequireEmptyLineBeforeBlockTagGroup", "SingleLineJavadoc", "SummaryJavadoc", "WriteTag"],
        "metrics": ["BooleanExpressionComplexity", "ClassDataAbstractionCoupling", "ClassFanOutComplexity", "CyclomaticComplexity", "JavaNCSS", "NPathComplexity"],
        "modifiers": ["ClassMemberImpliedModifier", "InterfaceMemberImpliedModifier", "ModifierOrder", "RedundantModifier"],
        "naming": ["AbbreviationAsWordInName", "AbstractClassName", "CatchParameterName", "ClassTypeParameterName", "ConstantName", "IllegalIdentifierName", "InterfaceTypeParameterName", "LambdaParameterName", "LocalFinalVariableName", "LocalVariableName", "MemberName", "MethodName", "MethodTypeParameterName", "PackageName", "ParameterName", "PatternVariableName", "RecordComponentName", "RecordTypeParameterName", "StaticVariableName", "TypeName"],
        "regexp": ["Regexp", "RegexpMultiline", "RegexpOnFilename", "RegexpSingleline", "RegexpSinglelineJava"],
        "whitespace": ["EmptyForInitializerPad", "EmptyForIteratorPad", "EmptyLineSeparator", "FileTabCharacter", "GenericWhitespace", "MethodParamPad", "NoLineWrap", "NoWhitespaceAfter", "NoWhitespaceBefore", "NoWhitespaceBeforeCaseDefaultColon", "OperatorWrap", "ParenPad", "SeparatorWrap", "SingleSpaceSeparator", "TypecastParenPad", "WhitespaceAfter", "WhitespaceAround"]
    }
    
    # Get all Java files
    java_files = [f for f in os.listdir(input_folder) if f.endswith(".java")]
    print(f"Found {len(java_files)} Java files to analyze with Checkstyle")
    
    for filename in java_files:
        file_path = os.path.join(input_folder, filename)
        # Extract task ID - handle both patterns: "taskid-X-genY.java" and simple "taskid.java"
        task_id = filename.replace(".java", "")
        
        result = {"Task ID": task_id}
        
        # Read the Java code
        try:
            with open(file_path, 'r') as f:
                result["Generated Code"] = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        output_file = os.path.join(checkstyle_output_dir, f"{task_id}_checkstyle.txt")
        
        # Run Checkstyle
        with open(output_file, 'w') as output:
            subprocess.run(
                ["java", "-jar", checkstyle_jar, "-c", config_file, file_path], 
                stdout=output, 
                stderr=subprocess.STDOUT
            )
        
        # Extract issues
        categorized_issues, file_error_counts = extract_checkstyle_issues(output_file, category_to_rules)
        
        for category, issues in categorized_issues.items():
            result[category] = "\n".join(issues)
            checkstyle_issues[category].extend(issues)
            
        for category, errors in file_error_counts.items():
            for rule, count in errors.items():
                error_code_counts[category][rule] += count
                
        results.append(result)
    
    # Save results
    if results:
        pd.DataFrame(results).to_excel(os.path.join(checkstyle_output_dir, 'checkstyle_analysis.xlsx'), index=False)
        generate_checkstyle_summary_json(checkstyle_output_dir, checkstyle_issues, error_code_counts, category_to_rules)
        print(f"Checkstyle analysis complete. Analyzed {len(results)} files.")
    else:
        print("No files were analyzed with Checkstyle.")

def generate_checkstyle_summary_json(output_dir, checkstyle_issues, error_code_counts, category_to_rules):
    total_files = len([f for f in os.listdir(output_dir) if f.endswith('_checkstyle.txt')])
    total_issues = {category: len(issues) for category, issues in checkstyle_issues.items()}
    formatted_error_counts = {category: dict(errors) for category, errors in error_code_counts.items()}
    
    for category in category_to_rules.keys():
        if category not in formatted_error_counts:
            formatted_error_counts[category] = {}
            
    summary = {
        "Total code count": {"files": total_files},
        "Total Rule Violations": total_issues,
        "Error Code Counts": formatted_error_counts
    }
    
    with open(os.path.join(output_dir, "checkstyle_summary.json"), 'w') as json_file:
        json.dump(summary, json_file, indent=4)

def run_pmd_analysis(input_folder, pmd_output_dir):
    os.makedirs(pmd_output_dir, exist_ok=True)
    categories = ["bestpractices", "codestyle", "design", "documentation", "errorprone", "multithreading", "performance", "security"]
    results = []
    all_counts = defaultdict(int)
    error_code_counts = defaultdict(lambda: defaultdict(int))
    processed_files = set()
    pmd_path = "/scratch/mhaque/pmd-bin-7.6.0/bin/pmd"
    
    # Get all Java files
    java_files = [f for f in os.listdir(input_folder) if f.endswith(".java")]
    print(f"Found {len(java_files)} Java files to analyze with PMD")
    
    for filename in java_files:
        file_path = os.path.join(input_folder, filename)
        # Extract task ID - handle both patterns: "taskid-X-genY.java" and simple "taskid.java"
        task_id = filename.replace(".java", "")
        processed_files.add(task_id)
        
        result = {"Task ID": task_id}
        
        # Read the Java code
        try:
            with open(file_path, 'r') as f:
                result["Generated Code"] = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        # Run PMD for each category
        for category in categories:
            output_file = os.path.join(pmd_output_dir, f"{task_id}_{category}.txt")
            
            # Run PMD analysis
            try:
                subprocess.run(
                    [pmd_path, "check", "-d", file_path, "-R", f"category/java/{category}.xml", "-f", "text", "-r", output_file],
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise exception on non-zero exit code
                )
            except Exception as e:
                print(f"Error running PMD for {filename} with category {category}: {e}")
                continue
            
            # Extract violations
            violations, violation_counts = extract_pmd_violations(output_file)
            
            # Store results
            if violations:
                result[category] = "\n".join(violations)
                all_counts[category] += len(violations)
                
                for violation, count in violation_counts.items():
                    error_code_counts[category][violation] += count
        
        results.append(result)
        print(f"Processed {task_id}")
    
    # Save results
    if results:
        pd.DataFrame(results).to_excel(os.path.join(pmd_output_dir, 'pmd_analysis.xlsx'), index=False)
        generate_pmd_summary_json(pmd_output_dir, len(processed_files), all_counts, error_code_counts)
        print(f"PMD analysis complete. Analyzed {len(processed_files)} files.")
    else:
        print("No files were analyzed with PMD.")

def generate_pmd_summary_json(output_dir, total_files, all_counts, error_code_counts):
    summary = {
        "Total code count": {"files": total_files},
        "Total Rule Violations": dict(all_counts),  # Convert defaultdict to dict
        "Error Code Counts": {category: dict(error_code_counts[category]) for category in all_counts}
    }
    
    with open(os.path.join(output_dir, "pmd_summary.json"), 'w') as json_file:
        json.dump(summary, json_file, indent=4)

def verify_java_files(input_folder):
    """Verify that Java files are properly formatted and can be processed."""
    java_files = [f for f in os.listdir(input_folder) if f.endswith(".java")]
    
    print(f"\n=== Java Files Verification ===")
    print(f"Total Java files found: {len(java_files)}")
    
    if java_files:
        print("\nSample file names:")
        for f in java_files[:5]:  # Show first 5 files
            print(f"  - {f}")
        
        # Check if files are readable
        sample_file = os.path.join(input_folder, java_files[0])
        try:
            with open(sample_file, 'r') as f:
                content = f.read()
                print(f"\nFirst file ({java_files[0]}) is readable.")
                print(f"File size: {len(content)} characters")
                print(f"First 200 characters:\n{content[:200]}...")
        except Exception as e:
            print(f"\nError reading sample file: {e}")
    else:
        print("\nNo Java files found! Please check:")
        print(f"  1. Input folder path: {input_folder}")
        print(f"  2. File extension (should be .java)")
        print(f"  3. Files in folder: {os.listdir(input_folder)[:5]}")
    
    return java_files

if __name__ == "__main__":
    # Update these paths to match your setup
    input_folder = "path/to/java_files"
    checkstyle_output_dir = "path/to/output/checkstyle"
    pmd_output_dir = "path/to/output/pmd"
    
    # Verify Java files first
    java_files = verify_java_files(input_folder)
    
    if java_files:
        # Uncomment the analysis you want to run
        # run_checkstyle_analysis(input_folder, checkstyle_output_dir)
        run_pmd_analysis(input_folder, pmd_output_dir)
    else:
        print("\nNo Java files to analyze. Please check your JSON-to-Java conversion.")
