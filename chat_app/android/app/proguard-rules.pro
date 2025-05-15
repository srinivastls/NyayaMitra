# Flutter-related rules
-keep class io.flutter.** { *; }
-keep class io.flutter.plugins.** { *; }
-dontwarn io.flutter.embedding.**

# Prevent obfuscating app's entry point
-keep class com.Legal_Chatbot.** { *; }
