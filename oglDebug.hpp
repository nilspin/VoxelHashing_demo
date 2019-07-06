/* A Callback function to log all GL errors.
 * To use, enable GL_DEBUG and subscribe this function to
 * glDebugMessageCallback() during app's initialization.
 * Stolen from here : https://stackoverflow.com/a/35052365/3925849
 */

#ifndef OGL_DEBUG_HPP
#define OGL_DEBUG_HPP

#include<GL/glu.h>
#include<iostream>

//Message callback
static void MessageCallback(GLenum source,
                     GLenum type,
                     GLuint id,
                     GLenum severity,
                     GLsizei length,
                     const GLchar* message,
                     const void* userParam)
{
  const char *_source = "Unknown";
  switch (source) {
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   _source = "WinSys";         break;
    case GL_DEBUG_SOURCE_APPLICATION:     _source = "App";            break;
    case GL_DEBUG_SOURCE_API:             _source = "OpenGL";         break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: _source = "ShaderCompiler"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:     _source = "3rdParty";       break;
    case GL_DEBUG_SOURCE_OTHER:           _source = "Other";          break;
  }
  const char *_type = "Unknown";
  switch (type) {
    case GL_DEBUG_TYPE_ERROR:               _type = "Error";       break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: _type = "Deprecated";  break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  _type = "Undefined";   break;
    case GL_DEBUG_TYPE_PORTABILITY:         _type = "Portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE:         _type = "Performance"; break;
    case GL_DEBUG_TYPE_MARKER:              _type = "Marker";      break;
    case GL_DEBUG_TYPE_PUSH_GROUP:          _type = "PushGrp";     break;
    case GL_DEBUG_TYPE_POP_GROUP:           _type = "PopGrp";      break;
    case GL_DEBUG_TYPE_OTHER:               _type = "Other";       break;
  }
  const char *_severity = "Unknown";
  switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:         _severity = "High";   break;
    case GL_DEBUG_SEVERITY_MEDIUM:       _severity = "Med";    break;
    case GL_DEBUG_SEVERITY_LOW:          _severity = "Low";    break;
    case GL_DEBUG_SEVERITY_NOTIFICATION: _severity = "Notify"; break;
  }
  std::cerr << _source << "." << _type << "[" << _severity << "](" <<
    id << "): " << message << std::endl;
}


#endif
