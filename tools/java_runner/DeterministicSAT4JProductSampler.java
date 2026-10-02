import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.sat4j.core.VecInt;
import org.sat4j.minisat.SolverFactory;
import org.sat4j.reader.DimacsReader;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.ISolver;

public class DeterministicSAT4JProductSampler {
    public static void main(String[] args) throws Exception {
        String modelPath = requireArg(args, 0, "model_path");
        String outputPath = requireArg(args, 1, "output_path");
        int targetProducts = Integer.parseInt(getArg(args, 2, "100"));
        long seed = Long.parseLong(getArg(args, 3, "0"));
        int assumptionCount = Integer.parseInt(getArg(args, 4, "64"));
        int maxAttemptsPerProduct = Integer.parseInt(getArg(args, 5, "10000"));

        if (targetProducts <= 0) {
            throw new IllegalArgumentException("target_products must be positive");
        }
        if (assumptionCount < 0) {
            throw new IllegalArgumentException("assumption_count must be non-negative");
        }
        if (maxAttemptsPerProduct <= 0) {
            throw new IllegalArgumentException("max_attempts_per_product must be positive");
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

        int numVariables = solver.nVars();
        int effectiveAssumptionCount = Math.min(assumptionCount, numVariables);
        Random random = new Random(seed);
        Set<String> fingerprints = new HashSet<String>();
        int attempts = 0;
        int unsatAttempts = 0;

        Path finalPath = Paths.get(outputPath);
        Path partialPath = Paths.get(outputPath + ".partial");
        Files.deleteIfExists(partialPath);
        PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(partialPath.toFile())));
        try {
            while (fingerprints.size() < targetProducts) {
                boolean found = false;
                for (int attempt = 0; attempt < maxAttemptsPerProduct; attempt++) {
                    attempts++;
                    VecInt assumptions = randomAssumptions(
                        numVariables,
                        effectiveAssumptionCount,
                        random
                    );
                    if (!solver.isSatisfiable(assumptions)) {
                        unsatAttempts++;
                        continue;
                    }

                    int[] product = completeModel(solver, numVariables);
                    String fingerprint = fingerprint(product);
                    if (!fingerprints.add(fingerprint)) {
                        continue;
                    }

                    writeProduct(writer, product);
                    if (fingerprints.size() < targetProducts) {
                        addBlockingClause(solver, product);
                    }
                    found = true;
                    break;
                }
                if (!found) {
                    throw new IllegalStateException(
                        "failed to find another unique product after "
                            + maxAttemptsPerProduct
                            + " attempts"
                    );
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
        System.out.println("num_variables=" + numVariables);
        System.out.println("target_products=" + targetProducts);
        System.out.println("unique_products=" + fingerprints.size());
        System.out.println("seed=" + seed);
        System.out.println("assumption_count=" + effectiveAssumptionCount);
        System.out.println("attempts=" + attempts);
        System.out.println("unsat_attempts=" + unsatAttempts);
    }

    private static VecInt randomAssumptions(int numVariables, int count, Random random) {
        VecInt assumptions = new VecInt(count);
        Set<Integer> selectedVariables = new HashSet<Integer>();
        while (selectedVariables.size() < count) {
            int variable = 1 + random.nextInt(numVariables);
            if (selectedVariables.add(variable)) {
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

    private static void writeProduct(PrintWriter writer, int[] product) {
        for (int index = 0; index < product.length; index++) {
            if (index > 0) {
                writer.print(';');
            }
            writer.print(product[index]);
        }
        writer.println();
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
