// claude made this file for organization before release
package org.network.util;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class AppFiles {
    private AppFiles() {}

    public static String resourceToTempFile(String resource) throws IOException {
        try (InputStream in = AppFiles.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new FileNotFoundException("Missing bundled resource: " + resource);
            }
            Path tmp = Files.createTempFile("fruit-", ".bin");
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            tmp.toFile().deleteOnExit();
            return tmp.toString();
        }
    }


    public static Path userDir(String subFolder) throws IOException {
        Path dir = Path.of(System.getProperty("user.home"), ".fruitdetector", subFolder);
        Files.createDirectories(dir);
        return dir;
    }
}