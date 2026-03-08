package org.pdf.TypeRule;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.GenericListVisitorAdapter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class JavaProgramAnalyzer {

    private static class NodeInfo {
        int id;
        String type;
        String label;
        Node astNode;
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("Usage: java -jar java-analyzer.jar <path-to-java-file>");
            System.exit(1);
        }

        File file = new File(args[0]);
        if (!file.exists()) {
            System.err.println("File not found: " + file.getAbsolutePath());
            System.exit(1);
        }

        String source = Files.readString(file.toPath());
        CompilationUnit cu = StaticJavaParser.parse(source);

        ObjectMapper mapper = new ObjectMapper();
        ObjectNode root = mapper.createObjectNode();

        root.put("file", file.getAbsolutePath());
        root.put("code", source);

        // AST
        ObjectNode astJson = buildAstJson(mapper, cu);
        root.set("ast", astJson);

        // CFG 
        ObjectNode cfgJson = buildSimpleCfgJson(mapper, cu);
        root.set("cfg", cfgJson);

        // DFG 
        ObjectNode dfgJson = buildSimpleDfgJson(mapper, cu);
        root.set("dfg", dfgJson);

        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(root));
    }

    private static ObjectNode buildAstJson(ObjectMapper mapper, CompilationUnit cu) {
        ObjectNode ast = mapper.createObjectNode();
        ArrayNode nodesArr = mapper.createArrayNode();
        ArrayNode edgesArr = mapper.createArrayNode();

        Map<Node, Integer> idMap = new IdentityHashMap<>();
        AtomicInteger counter = new AtomicInteger(0);

        // DFS по AST
        Deque<Node> stack = new ArrayDeque<>();
        stack.push(cu);
        idMap.put(cu, counter.getAndIncrement());

        while (!stack.isEmpty()) {
            Node current = stack.pop();
            int id = idMap.get(current);

            ObjectNode n = mapper.createObjectNode();
            n.put("id", id);
            n.put("type", current.getClass().getSimpleName());
            n.put("label", current.toString().replace("\n", " ").trim());
            nodesArr.add(n);

            for (Node child : current.getChildNodes()) {
                int childId = counter.getAndIncrement();
                idMap.put(child, childId);
                stack.push(child);

                ArrayNode e = mapper.createArrayNode();
                e.add(id);
                e.add(childId);
                edgesArr.add(e);
            }
        }

        ast.set("nodes", nodesArr);
        ast.set("edges", edgesArr);
        return ast;
    }

    private static ObjectNode buildSimpleCfgJson(ObjectMapper mapper, CompilationUnit cu) {
        ObjectNode cfg = mapper.createObjectNode();
        ArrayNode nodesArr = mapper.createArrayNode();
        ArrayNode edgesArr = mapper.createArrayNode();

        List<Statement> allStatements = cu.accept(new GenericListVisitorAdapter<Statement, Void>() {
            @Override
            public List<Statement> visit(MethodDeclaration n, Void arg) {
                List<Statement> res = new ArrayList<>();
                n.getBody().ifPresent(body -> res.addAll(body.getStatements()));
                return res;
            }
        }, null);

        for (int i = 0; i < allStatements.size(); i++) {
            Statement st = allStatements.get(i);
            ObjectNode n = mapper.createObjectNode();
            n.put("id", i);
            n.put("label", st.toString().replace("\n", " ").trim());
            nodesArr.add(n);

            if (i < allStatements.size() - 1) {
                ArrayNode e = mapper.createArrayNode();
                e.add(i);
                e.add(i + 1);
                edgesArr.add(e);
            }
        }

        cfg.set("nodes", nodesArr);
        cfg.set("edges", edgesArr);
        return cfg;
    }

    private static ObjectNode buildSimpleDfgJson(ObjectMapper mapper, CompilationUnit cu) {
        ObjectNode dfg = mapper.createObjectNode();
        ArrayNode nodesArr = mapper.createArrayNode();
        ArrayNode edgesArr = mapper.createArrayNode();

        // Узлы DFG будут "имя переменной + индекс появления"
        Map<String, List<Integer>> varOccurrences = new HashMap<>();
        AtomicInteger idCounter = new AtomicInteger(0);

        List<String> allLines = Arrays.asList(cu.toString().split("\n"));
        for (int i = 0; i < allLines.size(); i++) {
            String line = allLines.get(i);
            // очень грубый хак: ищем идентификаторы по regex \b[a-zA-Z_][a-zA-Z0-9_]*\b
            String[] tokens = line.split("[^a-zA-Z0-9_]");
            for (String t : tokens) {
                if (t.isBlank()) continue;
                if (Character.isJavaIdentifierStart(t.charAt(0))) {
                    varOccurrences.computeIfAbsent(t, k -> new ArrayList<>()).add(i);
                    ObjectNode n = mapper.createObjectNode();
                    int id = idCounter.getAndIncrement();
                    n.put("id", id);
                    n.put("var", t);
                    n.put("line", i);
                    nodesArr.add(n);
                }
            }
        }

        Map<String, List<Integer>> nodeIdsByVar = new HashMap<>();
        for (int i = 0; i < nodesArr.size(); i++) {
            String var = nodesArr.get(i).get("var").asText();
            nodeIdsByVar.computeIfAbsent(var, k -> new ArrayList<>()).add(nodesArr.get(i).get("id").asInt());
        }

        for (Map.Entry<String, List<Integer>> eVar : nodeIdsByVar.entrySet()) {
            List<Integer> ids = eVar.getValue();
            Collections.sort(ids);
            for (int i = 0; i < ids.size() - 1; i++) {
                ArrayNode e = mapper.createArrayNode();
                e.add(ids.get(i));
                e.add(ids.get(i + 1));
                edgesArr.add(e);
            }
        }

        dfg.set("nodes", nodesArr);
        dfg.set("edges", edgesArr);
        return dfg;
    }
}
