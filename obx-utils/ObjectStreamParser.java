import java.io.*;
import java.util.*;
import java.lang.reflect.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

public class ObjectStreamParser {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java ObjectStreamParser <objectStreamFile>");
            System.exit(1);
        }
        String filename = args[0];

        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            // If the file contains only one serialized object:
            Object obj = ois.readObject();

            Object jsonReady = convertToJson(obj);

            // Use Jackson to output pretty JSON:
            ObjectMapper mapper = new ObjectMapper();
            mapper.enable(SerializationFeature.INDENT_OUTPUT);
            String json = mapper.writeValueAsString(jsonReady);
            System.out.println(json);
        } catch (EOFException eof) {
        
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Recursively converts an object into a structure that can be serialized to JSON.
     * It tries to capture:
     * - Basic types directly.
     * - Arrays, Collections, and Maps by recursing on their elements.
     * - For standard objects, a map with keys "class" (the class name) and "fields" (a map of field names to converted values).
     */
    public static Object convertToJson(Object obj) throws IllegalAccessException {
        if (obj == null) {
            return null;
        }

        // Handle primitives, wrappers, strings, and enums:
        if (obj instanceof Number || obj instanceof String || obj instanceof Boolean ||
            obj instanceof Character || obj.getClass().isEnum()) {
            return obj;
        }

        // Handle arrays
        if (obj.getClass().isArray()) {
            int length = Array.getLength(obj);
            List<Object> list = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                Object arrayElem = Array.get(obj, i);
                list.add(convertToJson(arrayElem));
            }
            return list;
        }

        // Handle Collections (Lists, Sets, etc.)
        if (obj instanceof Collection) {
            Collection<?> col = (Collection<?>) obj;
            List<Object> list = new ArrayList<>(col.size());
            for (Object element : col) {
                list.add(convertToJson(element));
            }
            return list;
        }

        // Handle Maps
        if (obj instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) obj;
            Map<Object, Object> newMap = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                newMap.put(convertToJson(entry.getKey()), convertToJson(entry.getValue()));
            }
            return newMap;
        }

        // For any other object, use reflection to read its fields.
        Map<String, Object> result = new LinkedHashMap<>();
        // Save the class name as a value:
        result.put("class", obj.getClass().getName());

        Map<String, Object> fieldsMap = new LinkedHashMap<>();
        Field[] fields = obj.getClass().getDeclaredFields();
        for (Field field : fields) {
            // Optionally skip fields that belong to certain packages (e.g., java.awt)
            if (field.getDeclaringClass().getPackage() != null &&
                field.getDeclaringClass().getPackage().getName().startsWith("java.")) {
                continue;
            }
            field.setAccessible(true);
            Object fieldValue;
            try {
                fieldValue = field.get(obj);
            } catch (Exception e) {
                fieldValue = "Could not read field: " + e.getMessage();
            }
            fieldsMap.put(field.getName(), convertToJson(fieldValue));
        }
        result.put("fields", fieldsMap);

        return result;
    }
}
