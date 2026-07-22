package dev.mcbot.arena;

import org.bukkit.World;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class ArenaEnvironmentTest {
    @Test
    void enforcementRestoresPermanentClearMidday() {
        RecordingWorld recording = new RecordingWorld();

        MCAIPlugin.enforceArenaEnvironment(recording.world());

        assertEquals(6000L, recording.time);
        assertEquals(Boolean.FALSE, recording.storming);
        assertEquals(Boolean.FALSE, recording.thundering);
        assertEquals(Integer.MAX_VALUE, recording.weatherDuration);
        assertEquals(Integer.MAX_VALUE, recording.thunderDuration);
        assertEquals("false", recording.gameRules.get("doDaylightCycle"));
        assertEquals("false", recording.gameRules.get("doWeatherCycle"));
    }

    private static final class RecordingWorld implements InvocationHandler {
        private long time = Long.MIN_VALUE;
        private Boolean storming;
        private Boolean thundering;
        private int weatherDuration = Integer.MIN_VALUE;
        private int thunderDuration = Integer.MIN_VALUE;
        private final Map<String, String> gameRules = new HashMap<>();

        World world() {
            return (World) Proxy.newProxyInstance(
                    World.class.getClassLoader(), new Class<?>[] {World.class}, this);
        }

        @Override
        public Object invoke(Object proxy, Method method, Object[] arguments) {
            switch (method.getName()) {
                case "getTime":
                    return 18000L;
                case "hasStorm":
                case "isThundering":
                    return true;
                case "getWeatherDuration":
                case "getThunderDuration":
                    return 0;
                case "getGameRuleValue":
                    return "true";
                case "setTime":
                    time = (Long) arguments[0];
                    return null;
                case "setStorm":
                    storming = (Boolean) arguments[0];
                    return null;
                case "setThundering":
                    thundering = (Boolean) arguments[0];
                    return null;
                case "setWeatherDuration":
                    weatherDuration = (Integer) arguments[0];
                    return null;
                case "setThunderDuration":
                    thunderDuration = (Integer) arguments[0];
                    return null;
                case "setGameRuleValue":
                    gameRules.put((String) arguments[0], (String) arguments[1]);
                    return true;
                case "equals":
                    return proxy == arguments[0];
                case "hashCode":
                    return System.identityHashCode(proxy);
                case "toString":
                    return "RecordingWorld";
                default:
                    return defaultValue(method.getReturnType());
            }
        }

        private static Object defaultValue(Class<?> type) {
            if (!type.isPrimitive()) return null;
            if (type == boolean.class) return false;
            if (type == byte.class) return (byte) 0;
            if (type == short.class) return (short) 0;
            if (type == int.class) return 0;
            if (type == long.class) return 0L;
            if (type == float.class) return 0.0F;
            if (type == double.class) return 0.0D;
            if (type == char.class) return '\0';
            return null;
        }
    }
}
