package org.pdf.TypeRule;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.Name;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithAnnotations;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.Statement;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

public class JavaProgramAnalyzer {

    private static final class AstBuildResult {
        final ObjectNode astJson;
        final Map<Node, Integer> astIdMap;

        AstBuildResult(ObjectNode astJson, Map<Node, Integer> astIdMap) {
            this.astJson = astJson;
            this.astIdMap = astIdMap;
        }
    }

    private static final class DefInfo {
        final int dfgNodeId;
        final String symbolKey;
        final String variable;
        final String method;
        final Node astNode;
        final int line;

        DefInfo(int dfgNodeId, String symbolKey, String variable, String method, Node astNode, int line) {
            this.dfgNodeId = dfgNodeId;
            this.symbolKey = symbolKey;
            this.variable = variable;
            this.method = method;
            this.astNode = astNode;
            this.line = line;
        }
    }

    private static final class DfgContext {
        final Map<String, DefInfo> currentDefs = new LinkedHashMap<>();
        final AtomicInteger nextNodeId;
        final ArrayNode nodesArr;
        final ArrayNode edgesArr;
        final ObjectMapper mapper;
        final Map<Node, Integer> astIdMap;
        final String methodName;

        DfgContext(ArrayNode nodesArr, ArrayNode edgesArr, ObjectMapper mapper,
                   Map<Node, Integer> astIdMap, String methodName, AtomicInteger nextNodeId) {
            this.nextNodeId = nextNodeId;
            this.nodesArr = nodesArr;
            this.edgesArr = edgesArr;
            this.mapper = mapper;
            this.astIdMap = astIdMap;
            this.methodName = methodName;
        }
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

        AstBuildResult astBuild = buildAstJson(mapper, cu);
        root.set("ast", astBuild.astJson);
        root.set("cfg", buildSimpleCfgJson(mapper, cu, astBuild.astIdMap));
        root.set("dfg", buildStructuredDfgJson(mapper, cu, astBuild.astIdMap));

        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(root));
    }

    private static AstBuildResult buildAstJson(ObjectMapper mapper, CompilationUnit cu) {
        ObjectNode ast = mapper.createObjectNode();
        ArrayNode nodesArr = mapper.createArrayNode();
        ArrayNode edgesArr = mapper.createArrayNode();

        Map<Node, Integer> idMap = new IdentityHashMap<>();
        AtomicInteger nextId = new AtomicInteger(0);
        assignAstIdsPreorder(cu, idMap, nextId);
        emitAstPreorder(cu, mapper, nodesArr, edgesArr, idMap);

        ast.set("nodes", nodesArr);
        ast.set("edges", edgesArr);
        return new AstBuildResult(ast, idMap);
    }

    private static void assignAstIdsPreorder(Node node, Map<Node, Integer> idMap, AtomicInteger nextId) {
        idMap.put(node, nextId.getAndIncrement());
        List<Node> children = new ArrayList<>(node.getChildNodes());
        children.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (Node child : children) {
            assignAstIdsPreorder(child, idMap, nextId);
        }
    }

    private static void emitAstPreorder(Node node, ObjectMapper mapper, ArrayNode nodesArr,
                                        ArrayNode edgesArr, Map<Node, Integer> idMap) {
        ObjectNode n = mapper.createObjectNode();
        n.put("id", idMap.get(node));
        n.put("type", node.getClass().getSimpleName());
        n.put("label", cleanText(node.toString()));
        addSourceSpan(n, node);
        n.put("method", enclosingMethodName(node));
        nodesArr.add(n);

        List<Node> children = new ArrayList<>(node.getChildNodes());
        children.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (Node child : children) {
            ArrayNode e = mapper.createArrayNode();
            e.add(idMap.get(node));
            e.add(idMap.get(child));
            edgesArr.add(e);
            emitAstPreorder(child, mapper, nodesArr, edgesArr, idMap);
        }
    }

    private static ObjectNode buildSimpleCfgJson(ObjectMapper mapper, CompilationUnit cu, Map<Node, Integer> astIdMap) {
        ObjectNode cfg = mapper.createObjectNode();
        ArrayNode nodesArr = mapper.createArrayNode();
        ArrayNode edgesArr = mapper.createArrayNode();

        AtomicInteger nextId = new AtomicInteger(0);
        for (MethodDeclaration method : cu.findAll(MethodDeclaration.class)) {
            if (method.getBody().isEmpty()) {
                continue;
            }
            List<Statement> stmts = new ArrayList<>(method.getBody().get().getStatements());
            stmts.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
            List<Integer> cfgIdsInMethod = new ArrayList<>();
            for (Statement st : stmts) {
                int cfgId = nextId.getAndIncrement();
                cfgIdsInMethod.add(cfgId);
                ObjectNode n = mapper.createObjectNode();
                n.put("id", cfgId);
                n.put("label", cleanText(st.toString()));
                n.put("method", method.getNameAsString());
                addSourceSpan(n, st);
                Integer astId = astIdMap.get(st);
                if (astId != null) {
                    n.put("ast_id", astId);
                }
                nodesArr.add(n);
            }
            for (int i = 0; i < cfgIdsInMethod.size() - 1; i++) {
                ObjectNode e = mapper.createObjectNode();
                e.put("src", cfgIdsInMethod.get(i));
                e.put("dst", cfgIdsInMethod.get(i + 1));
                e.put("type", "next_stmt");
                edgesArr.add(e);
            }
        }

        cfg.set("nodes", nodesArr);
        cfg.set("edges", edgesArr);
        return cfg;
    }

    private static ObjectNode buildStructuredDfgJson(ObjectMapper mapper, CompilationUnit cu, Map<Node, Integer> astIdMap) {
        ObjectNode dfg = mapper.createObjectNode();
        ArrayNode nodesArr = mapper.createArrayNode();
        ArrayNode edgesArr = mapper.createArrayNode();

        AtomicInteger globalNodeIds = new AtomicInteger(0);
        for (MethodDeclaration method : cu.findAll(MethodDeclaration.class)) {
            DfgContext ctx = new DfgContext(nodesArr, edgesArr, mapper, astIdMap, method.getNameAsString(), globalNodeIds);

            for (Parameter p : method.getParameters()) {
                DefInfo def = addDefNode(ctx, p.getNameAsString(), p, "param_def");
                ctx.currentDefs.put(p.getNameAsString(), def);
            }

            for (Parameter p : method.getParameters()) {
                collectAnnotationRefs(ctx, p, p.getAnnotations(), p.getNameAsString());
            }

            if (method.getBody().isPresent()) {
                processBlock(ctx, method.getBody().get());
            }
        }

        dfg.set("nodes", nodesArr);
        dfg.set("edges", edgesArr);
        return dfg;
    }

    private static void processBlock(DfgContext ctx, BlockStmt block) {
        List<Statement> statements = new ArrayList<>(block.getStatements());
        statements.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (Statement stmt : statements) {
            processStatement(ctx, stmt);
        }
    }

    private static void processStatement(DfgContext ctx, Statement stmt) {
        if (stmt.isBlockStmt()) {
            processBlock(ctx, stmt.asBlockStmt());
            return;
        }

        List<VariableDeclarator> decls = stmt.findAll(VariableDeclarator.class);
        decls.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (VariableDeclarator vd : decls) {
            processVariableDeclarator(ctx, vd);
        }

        List<AssignExpr> assigns = stmt.findAll(AssignExpr.class);
        assigns.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (AssignExpr assign : assigns) {
            processAssignExpr(ctx, assign);
        }

        // Standalone method calls and other expressions still contribute uses.
        if (!stmt.isExpressionStmt()) {
            collectExpressionUses(ctx, stmt, "stmt_use");
        }
    }

    private static void processVariableDeclarator(DfgContext ctx, VariableDeclarator vd) {
        if (vd.getInitializer().isPresent()) {
            collectExpressionUses(ctx, vd.getInitializer().get(), "rhs_use");
        }
        DefInfo def = addDefNode(ctx, vd.getNameAsString(), vd, "local_def");
        ctx.currentDefs.put(vd.getNameAsString(), def);

        Node parent = vd.getParentNode().orElse(null);
        if (parent instanceof NodeWithAnnotations<?> annotatable) {
            collectAnnotationRefs(ctx, parent, annotatable.getAnnotations(), vd.getNameAsString());
        }
    }

    private static void processAssignExpr(DfgContext ctx, AssignExpr assign) {
        collectExpressionUses(ctx, assign.getValue(), "rhs_use");
        if (assign.getTarget().isNameExpr()) {
            NameExpr target = assign.getTarget().asNameExpr();
            DefInfo def = addDefNode(ctx, target.getNameAsString(), assign, "assign_def");
            ctx.currentDefs.put(target.getNameAsString(), def);
        } else {
            collectExpressionUses(ctx, assign.getTarget(), "lhs_complex_use");
        }
    }

    private static void collectExpressionUses(DfgContext ctx, Node root, String role) {
        if (root == null) {
            return;
        }
        List<NameExpr> nameExprs = new ArrayList<>(root.findAll(NameExpr.class));
        nameExprs.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (NameExpr nameExpr : nameExprs) {
            String var = nameExpr.getNameAsString();
            DefInfo def = ctx.currentDefs.get(var);
            if (def == null) {
                continue;
            }
            addUseNodeAndEdge(ctx, def, var, nameExpr, role, "data_flow", nameExpr.toString(), null);
        }

        List<MethodCallExpr> calls = new ArrayList<>(root.findAll(MethodCallExpr.class));
        calls.sort(Comparator.comparing(JavaProgramAnalyzer::nodeStart));
        for (MethodCallExpr call : calls) {
            if (call.getScope().isPresent() && call.getScope().get().isNameExpr()) {
                NameExpr scope = call.getScope().get().asNameExpr();
                DefInfo def = ctx.currentDefs.get(scope.getNameAsString());
                if (def != null) {
                    addUseNodeAndEdge(ctx, def, scope.getNameAsString(), scope, role,
                            "receiver_use", call.toString(), null);
                }
            }
        }
    }

    private static void collectAnnotationRefs(DfgContext ctx, Node ownerNode, List<AnnotationExpr> annotations, String ownerVar) {
        for (AnnotationExpr ann : annotations) {
            if (ann.isNormalAnnotationExpr()) {
                NormalAnnotationExpr normal = ann.asNormalAnnotationExpr();
                normal.getPairs().forEach(pair -> {
                    Expression value = pair.getValue();
                    if (value.isStringLiteralExpr()) {
                        String refName = value.asStringLiteralExpr().asString();
                        maybeAddAnnotationRef(ctx, ann, ownerNode, ownerVar, pair.getNameAsString(), refName);
                    }
                });
            } else if (ann.isSingleMemberAnnotationExpr()) {
                SingleMemberAnnotationExpr single = ann.asSingleMemberAnnotationExpr();
                if (single.getMemberValue().isStringLiteralExpr()) {
                    String refName = single.getMemberValue().asStringLiteralExpr().asString();
                    maybeAddAnnotationRef(ctx, ann, ownerNode, ownerVar, "value", refName);
                }
            }
        }
    }

    private static void maybeAddAnnotationRef(DfgContext ctx, AnnotationExpr annotation, Node ownerNode,
                                              String ownerVar, String paramName, String referencedName) {
        if (referencedName == null || referencedName.isBlank()) {
            return;
        }
        DefInfo def = ctx.currentDefs.get(referencedName);
        if (def == null) {
            return;
        }
        Map<String, String> extra = new LinkedHashMap<>();
        extra.put("annotation", cleanText(annotation.toString()));
        extra.put("annotation_param", paramName);
        extra.put("owner_var", ownerVar == null ? "" : ownerVar);
        if (ownerNode != null) {
            Integer ownerAstId = ctx.astIdMap.get(ownerNode);
            if (ownerAstId != null) {
                extra.put("owner_ast_id", Integer.toString(ownerAstId));
            }
            extra.put("owner_type", ownerNode.getClass().getSimpleName());
        }
        addUseNodeAndEdge(ctx, def, referencedName, annotation, "annotation_ref", "annotation_ref", annotation.toString(), extra);
    }

    private static DefInfo addDefNode(DfgContext ctx, String variable, Node astNode, String role) {
        int id = ctx.nextNodeId.getAndIncrement();
        int line = nodeLine(astNode);
        String symbolKey = ctx.methodName + ":" + variable + "@" + line;

        ObjectNode n = ctx.mapper.createObjectNode();
        n.put("id", id);
        n.put("var", variable);
        n.put("symbol", symbolKey);
        n.put("method", ctx.methodName);
        n.put("role", role);
        n.put("label", cleanText(astNode.toString()));
        addSourceSpan(n, astNode);
        Integer astId = ctx.astIdMap.get(astNode);
        if (astId != null) {
            n.put("ast_id", astId);
        }
        ctx.nodesArr.add(n);

        return new DefInfo(id, symbolKey, variable, ctx.methodName, astNode, line);
    }

    private static void addUseNodeAndEdge(DfgContext ctx, DefInfo def, String variable, Node useNode,
                                          String role, String edgeType, String label,
                                          Map<String, String> extraFields) {
        int id = ctx.nextNodeId.getAndIncrement();
        ObjectNode n = ctx.mapper.createObjectNode();
        n.put("id", id);
        n.put("var", variable);
        n.put("symbol", def.symbolKey);
        n.put("method", ctx.methodName);
        n.put("role", role);
        n.put("label", cleanText(label));
        addSourceSpan(n, useNode);
        Integer astId = ctx.astIdMap.get(useNode);
        if (astId != null) {
            n.put("ast_id", astId);
        }
        if (extraFields != null) {
            extraFields.forEach(n::put);
        }
        ctx.nodesArr.add(n);

        ObjectNode e = ctx.mapper.createObjectNode();
        e.put("src", def.dfgNodeId);
        e.put("dst", id);
        e.put("type", edgeType);
        e.put("var", variable);
        e.put("symbol", def.symbolKey);
        e.put("method", ctx.methodName);
        e.put("src_line", def.line);
        e.put("dst_line", nodeLine(useNode));
        ctx.edgesArr.add(e);
    }

    private static void addSourceSpan(ObjectNode out, Node node) {
        Optional<Range> maybeRange = node.getRange();
        if (maybeRange.isPresent()) {
            Range range = maybeRange.get();
            out.put("line", range.begin.line);
            out.put("column", range.begin.column);
            out.put("end_line", range.end.line);
            out.put("end_column", range.end.column);
        } else {
            out.putNull("line");
            out.putNull("column");
            out.putNull("end_line");
            out.putNull("end_column");
        }
    }

    private static int nodeLine(Node node) {
        return node.getRange().map(r -> r.begin.line).orElse(-1);
    }

    private static Position nodeStart(Node node) {
        return node.getRange().map(r -> r.begin).orElse(new Position(Integer.MAX_VALUE, Integer.MAX_VALUE));
    }

    private static String enclosingMethodName(Node node) {
        if (node instanceof MethodDeclaration) {
            return ((MethodDeclaration) node).getNameAsString();
        }
        return node.findAncestor(MethodDeclaration.class)
                .map(MethodDeclaration::getNameAsString)
                .orElse("<global>");
    }

    private static String cleanText(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\r", " ").replace("\n", " ").replaceAll("\\s+", " ").trim();
    }
}
