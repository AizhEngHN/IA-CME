import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.sat4j.core.VecInt;
import org.sat4j.minisat.SolverFactory;
import org.sat4j.minisat.core.Solver;
import org.sat4j.minisat.orders.RandomLiteralSelectionStrategy;
import org.sat4j.minisat.orders.RandomWalkDecorator;
import org.sat4j.minisat.orders.VarOrderHeap;
import org.sat4j.reader.DimacsReader;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.ISolver;

public class TargetedSAT4JProductGenerator {
    public static void main(String[] args) throws Exception {
        String modelPath = requireArg(args, 0, "model_path");
        String targetsPath = requireArg(args, 1, "targets_path");
        String outputPath = requireArg(args, 2, "output_path");
        long seed = Long.parseLong(getArg(args, 3, "0"));
        int randomAssumptionCount = Integer.parseInt(getArg(args, 4, "8"));
        int maxAttemptsPerTarget = Integer.parseInt(getArg(args, 5, "1000"));
        boolean randomizeSolverOrder = Boolean.parseBoolean(getArg(args, 6, "false"));
        if (randomAssumptionCount < 0 || maxAttemptsPerTarget <= 0) {
            throw new IllegalArgumentException("invalid random assumption or attempt count");
        }

        ISolver solver = SolverFactory.newDefault();
        solver.setTimeout(1000);
        DimacsReader reader = new DimacsReader(solver);
        FileInputStream input = new FileInputStream(modelPath);
        try {
            reader.parseInstance(input);
        } finally {
            input.close();
        }

        List<int[]> targets = readTargets(targetsPath, solver.nVars());
        Random random = new Random(seed);
        if (randomizeSolverOrder) {
            seedSolverRandomSources(seed);
        }
        Set<String> fingerprints = new HashSet<String>();
        int attempts = 0;
        int unsatAttempts = 0;
        int fallbackAttempts = 0;
        int failures = 0;

        Path finalPath = Paths.get(outputPath);
        Path partialPath = Paths.get(outputPath + ".partial");
        Files.deleteIfExists(partialPath);
        PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(partialPath.toFile())));
        try {
            for (int targetIndex = 0; targetIndex < targets.size(); targetIndex++) {
                int[] target = targets.get(targetIndex);
                int[] product = null;
                for (int attempt = 0; attempt < maxAttemptsPerTarget; attempt++) {
                    attempts++;
                    VecInt assumptions = targetedAssumptions(
                        target,
                        solver.nVars(),
                        randomAssumptionCount,
                        random
                    );
                    if (randomizeSolverOrder) {
                        ((Solver) solver).setOrder(
                            new RandomWalkDecorator(
                                new VarOrderHeap(new RandomLiteralSelectionStrategy()),
                                1
                            )
                        );
                    }
                    if (!solver.isSatisfiable(assumptions)) {
                        unsatAttempts++;
                        continue;
                    }
                    int[] candidate = completeModel(solver, solver.nVars());
                    if (fingerprints.add(fingerprint(candidate))) {
                        product = candidate;
                        break;
                    }
                }

                if (product == null) {
                    fallbackAttempts++;
                    if (solver.isSatisfiable(new VecInt(target))) {
                        int[] candidate = completeModel(solver, solver.nVars());
                        if (fingerprints.add(fingerprint(candidate))) {
                            product = candidate;
                        }
                    }
                }

                if (product == null) {
                    failures++;
                    continue;
                }
                writeTargetedProduct(writer, targetIndex, product);
                if (targetIndex + 1 < targets.size()) {
                    addBlockingClause(solver, product);
                }
            }
        } catch (Exception error) {
            writer.close();
            Files.deleteIfExists(partialPath);
            throw error;
        } finally {
            writer.close();
        }
        moveCompletedOutput(partialPath, finalPath);

        System.out.println("status=complete");
        System.out.println("num_variables=" + solver.nVars());
        System.out.println("target_requests=" + targets.size());
        System.out.println("generated_products=" + fingerprints.size());
        System.out.println("seed=" + seed);
        System.out.println("random_assumption_count=" + randomAssumptionCount);
        System.out.println("randomize_solver_order=" + randomizeSolverOrder);
        System.out.println("solver_order_seed=" + (randomizeSolverOrder ? Long.toString(seed) : "disabled"));
        System.out.println("attempts=" + attempts);
        System.out.println("unsat_attempts=" + unsatAttempts);
        System.out.println("fallback_attempts=" + fallbackAttempts);
        System.out.println("target_failures=" + failures);
    }

    private static List<int[]> readTargets(String path, int numVariables) throws Exception {
        List<int[]> targets = new ArrayList<int[]>();
        BufferedReader reader = new BufferedReader(new FileReader(path));
        try {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] tokens = line.split(";");
                int[] target = new int[tokens.length];
                Set<Integer> variables = new HashSet<Integer>();
                for (int index = 0; index < tokens.length; index++) {
                    int literal = Integer.parseInt(tokens[index].trim());
                    int variable = Math.abs(literal);
                    if (literal == 0 || variable > numVariables || !variables.add(variable)) {
                        throw new IllegalArgumentException("invalid target interaction: " + line);
                    }
                    target[index] = literal;
                }
                targets.add(target);
            }
        } finally {
            reader.close();
        }
        if (targets.isEmpty()) {
            throw new IllegalArgumentException("target file is empty");
        }
        return targets;
    }

    private static VecInt targetedAssumptions(
        int[] target,
        int numVariables,
        int randomCount,
        Random random
    ) {
        Set<Integer> usedVariables = new HashSet<Integer>();
        VecInt assumptions = new VecInt(target.length + randomCount);
        for (int literal : target) {
            assumptions.push(literal);
            usedVariables.add(Math.abs(literal));
        }
        int effectiveCount = Math.min(randomCount, numVariables - usedVariables.size());
        while (usedVariables.size() < target.length + effectiveCount) {
            int variable = 1 + random.nextInt(numVariables);
            if (usedVariables.add(variable)) {
                assumptions.push(random.nextBoolean() ? variable : -variable);
            }
        }
        return assumptions;
    }

    private static int[] completeModel(ISolver solver, int numVariables) {
        int[] product = new int[numVariables];
        for (int variable = 1; variable <= numVariables; variable++) {
            product[variable - 1] = solver.model(variable) ? variable : -variable;
        }
        return product;
    }

    private static void addBlockingClause(ISolver solver, int[] product) throws ContradictionException {
        int[] blocking = new int[product.length];
        for (int index = 0; index < product.length; index++) {
            blocking[index] = -product[index];
        }
        solver.addBlockingClause(new VecInt(blocking));
    }

    private static String fingerprint(int[] product) {
        StringBuilder builder = new StringBuilder(product.length);
        for (int value : product) {
            builder.append(value > 0 ? '1' : '0');
        }
        return builder.toString();
    }

    private static void writeTargetedProduct(PrintWriter writer, int targetIndex, int[] product) {
        writer.print(targetIndex);
        writer.print('\t');
        for (int index = 0; index < product.length; index++) {
            if (index > 0) {
                writer.print(';');
            }
            writer.print(product[index]);
        }
        writer.println();
    }

    private static void seedSolverRandomSources(long seed) throws Exception {
        RandomLiteralSelectionStrategy.RAND.setSeed(seed ^ 0x6A09E667F3BCC909L);
        Field randomWalkField = RandomWalkDecorator.class.getDeclaredField("rand");
        randomWalkField.setAccessible(true);
        Random randomWalk = (Random) randomWalkField.get(null);
        randomWalk.setSeed(seed ^ 0xBB67AE8584CAA73BL);
    }

    private static void moveCompletedOutput(Path partialPath, Path finalPath) throws Exception {
        try {
            Files.move(
                partialPath,
                finalPath,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            );
        } catch (AtomicMoveNotSupportedException error) {
            Files.move(partialPath, finalPath, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String requireArg(String[] args, int index, String name) {
        if (args.length <= index || args[index] == null || args[index].trim().isEmpty()) {
            throw new IllegalArgumentException("missing required argument: " + name);
        }
        return args[index].trim();
    }

    private static String getArg(String[] args, int index, String defaultValue) {
        if (args.length <= index || args[index] == null || args[index].trim().isEmpty()) {
            return defaultValue;
        }
        return args[index].trim();
    }
}
