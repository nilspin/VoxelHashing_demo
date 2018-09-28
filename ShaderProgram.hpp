/***
author     : r3dux
version    : 0.3 - 15/01/2014
description: Gets GLSL source code either provided as strings or can load from filenames,
compiles the shaders, creates a shader program which the shaders are linked
to, then the program is validated and is ready for use via myProgram.use(),
<draw-stuff-here> then calling myProgram.disable();

Attributes and uniforms are stored in <string, int> maps and can be added
via calls to addAttribute(<name-of-attribute>) and then the attribute
index can be obtained via myProgram.attribute(<name-of-attribute>) - Uniforms
work in the exact same way.

Edit: Extended support for tess, geometry and compute shaders.
Shader text files with extensions ".vert" ".frag" ".geom" ".eval"(tess eval) ".cont"(tess control)
and ".comp"(compute shader) will be loaded accordingly.

Edited by nilspin - 19/3/2018.
***/

#ifndef SHADER_PROGRAM_HPP
#define SHADER_PROGRAM_HPP

#include "opengl_win.h"

#include<boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>


class ShaderProgram
{
private:
	//For use with templates
	std::vector<std::string> fileNameContainer;

	// static DEBUG flag - if set to false then, errors aside, we'll run completely silent
	static const bool DEBUG = true;

	// We'll use an enum to differentiate between shaders and shader programs when querying the info log
	enum class ObjectType
	{
		SHADER, PROGRAM
	};

	// Shader program and individual shader Ids
	GLuint programId;
	
	std::map<std::string, GLenum> shaderEnumMap
	{
		{".vert", GL_VERTEX_SHADER},
		{".frag", GL_FRAGMENT_SHADER},
		{".geom", GL_GEOMETRY_SHADER},
		{".eval", GL_TESS_EVALUATION_SHADER},
		{".cont", GL_TESS_CONTROL_SHADER},
		{".comp", GL_COMPUTE_SHADER}
	};

	// How many shaders are attached to the shader program
	GLuint shaderCount;

	// Map of attributes and their binding locations
	std::map<std::string, int> attributeMap;

	// Map of uniforms and their binding locations
	std::map<std::string, int> uniformMap;

	// Has this shader program been initialised?
	bool initialised;

	// ---------- PRIVATE METHODS ----------

	// Private method to compile/attach/link/verify the shaders.
	// Note: Rather than returning a boolean as a success/fail status we'll just consider
	// a failure here to be an unrecoverable error and throw a runtime_error.
	void initialise(std::vector<std::string> fileList)
	{
		std::vector<GLuint> loadedShaders;
		for(auto fileName:fileList)
		{
			/*We need boost for just this one line.
			TODO: Replace with std::filesystem when compiler support is available*/
			boost::filesystem::path fname{fileName};
			std::string extension = fname.extension().string();
			GLenum shaderType = shaderEnumMap.find(extension)->second;

			std::string shaderSource = loadShaderFromFile(fileName);

			// Compile the shaders and return their id values
			GLuint shaderId = compileShader(shaderSource, shaderType);
			
			// Attach the compiled shader to the shader program
			glAttachShader(programId, shaderId);

			loadedShaders.push_back(shaderId);
			
		}
		
		// Link the shader program - details are placed in the program info log
		glLinkProgram(programId);
		
		// Once the shader program has the shaders attached and linked, the shaders are no longer required.
		// If the linking failed, then we're going to abort anyway so we still detach the shaders.
		for(auto shader:loadedShaders)
		{
			glDetachShader(programId, shader);
		}
		
		checkProgramLinkStatus();
		
		// Validate the shader program
		glValidateProgram(programId);

		checkProgramValidationStatus();
		
		// Finally, the shader program is initialised
		initialised=true;
	}

	// Private method to compile a shader of a given type
	GLuint compileShader(std::string shaderSource, GLenum shaderType)
	{
		std::string shaderTypeString = std::to_string(shaderType);
		
		// Generate a shader id
		// Note: Shader id will be non-zero if successfully created.
		GLuint shaderId = glCreateShader(shaderType);
		if (shaderId == 0)
		{
			// Display the shader log via a runtime_error
			throw std::runtime_error("Could not create shader of type " + shaderTypeString + ": " + getInfoLog(ObjectType::SHADER, shaderId));
		}

		// Get the source string as a pointer to an array of characters
		const char *shaderSourceChars = shaderSource.c_str();

		// Attach the GLSL source code to the shader
		// Params: GLuint shader, GLsizei count, const GLchar **string, const GLint *length
		// Note: The pointer to an array of source chars will be null terminated, so we don't need to specify the length and can instead use NULL.
		glShaderSource(shaderId, 1, &shaderSourceChars, NULL);

		// Compile the shader
		glCompileShader(shaderId);

		checkCompileStatus(shaderId, shaderTypeString);

		// If everything went well, return the shader id
		return shaderId;
	}
	
	// Check the compilation status and throw a runtime_error if shader compilation failed
	void checkCompileStatus(GLint shaderId, std::string shaderTypeString)
	{
		GLint shaderStatus;
		glGetShaderiv(shaderId, GL_COMPILE_STATUS, &shaderStatus);
		if (shaderStatus == GL_FALSE)
		{
			GLint maxLength = 0;
			glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &maxLength);

			// The maxLength includes the NULL character
			GLchar* errorLog = (GLchar*)malloc(maxLength);
			glGetShaderInfoLog(shaderId, maxLength, &maxLength, errorLog);
			
			std::cout << "Shader compile error : " << errorLog << std::endl;
			free(errorLog);
			glDeleteShader(shaderId); // Don't leak the shader.
			throw std::runtime_error(shaderTypeString + " compilation failed: " + getInfoLog(ObjectType::SHADER, shaderId));
		}
		else
		{
			if (DEBUG)
			{
				std::cout << shaderTypeString << " shader compilation successful." << std::endl;
			}
		}
	}

	// Check the validation status and throw a runtime_error if program validation failed
	void checkProgramValidationStatus()
	{
		GLint programValidatationStatus;
		glGetProgramiv(programId, GL_VALIDATE_STATUS, &programValidatationStatus);
		if (programValidatationStatus == GL_TRUE)
		{
			if (DEBUG)
			{
				std::cout << "Shader program validation successful." << std::endl;
			}
		}
		else
		{
			throw std::runtime_error("Shader program validation failed: " + getInfoLog(ObjectType::PROGRAM, programId));
		}
	}

	void checkProgramLinkStatus()
	{
		// Check the program link status and throw a runtime_error if program linkage failed.
		GLint programLinkSuccess = GL_FALSE;
		glGetProgramiv(programId, GL_LINK_STATUS, &programLinkSuccess);
		if (programLinkSuccess == GL_TRUE)
		{
			if (DEBUG)
			{
				std::cout << "Shader program link successful." << std::endl;
			}
		}
		else
		{
			GLint maxLength = 0;
			glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &maxLength);

			//The maxLength includes the NULL character
			GLchar *log = (GLchar *)malloc(maxLength);
			glGetProgramInfoLog(programId, maxLength, 0, log); 
			std::cout << "Linker error : " << log << std::endl;
			free(log);
			//The program is useless now. So delete it.
			glDeleteProgram(programId);

			throw std::runtime_error("Shader program link failed: " + getInfoLog(ObjectType::PROGRAM, programId));
		}
	}
	// Private method to load the shader source code from a file
	std::string loadShaderFromFile(const std::string filename)
	{
		// Create an input filestream and attempt to open the specified file
		std::ifstream file(filename.c_str());

		// If we couldn't open the file we'll bail out
		if (!file.good())
		{
			throw std::runtime_error("Failed to open file: " + filename);
		}

		// Otherwise, create a string stream...
		std::stringstream stream;

		// ...and dump the contents of the file into it.
		stream << file.rdbuf();

		// Now that we've read the file we can close it
		file.close();

		// Finally, convert the stringstream into a string and return it
		return stream.str();
	}

	// Private method to return the current shader program info log as a string
	std::string getInfoLog(ObjectType type, int id)
	{
		GLint infoLogLength;
		if (type == ObjectType::SHADER)
		{
			glGetShaderiv(id, GL_INFO_LOG_LENGTH, &infoLogLength);
		}
		else // type must be ObjectType::PROGRAM
		{
			glGetProgramiv(id, GL_INFO_LOG_LENGTH, &infoLogLength);
		}

		GLchar *infoLog = new GLchar[infoLogLength + 1];
		if (type == ObjectType::SHADER)
		{
			glGetShaderInfoLog(id, infoLogLength, NULL, infoLog);
		}
		else // type must be ObjectType::PROGRAM
		{
			glGetProgramInfoLog(id, infoLogLength, NULL, infoLog);
		}

		// Convert the info log to a string
		std::string infoLogString(infoLog);

		// Delete the char array version of the log
		delete[] infoLog;

		// Finally, return the string version of the info log
		return infoLogString;
	}

public:
	/*--------------------TEMPLATE CODE--------------------------------*/
	//Simple initializer for string
	template<typename T>
	std::string to_string(const T t)
	{
		return t;
	}

	/*
	Converts arguments into vector of string.
	Converted args are put into initializer list
	(which compiler resolves by itself) and initialised 
	into a vector.
	*/
	template<typename ... Params>
	void initFromFiles(const Params& ... args)
	{
		std::vector<std::string> temp {to_string(args)...};
		fileNameContainer = temp;
		initialise(fileNameContainer);
	}
	/*-----------------------------------------------------------------*/


	// Constructor
	ShaderProgram()
	{
		// We start in a non-initialised state - calling initFromFiles() or initFromStrings() will
		// initialise us.
		initialised = false;

		// Generate a unique Id / handle for the shader program
		// Note: We MUST have a valid rendering context before generating the programId or we'll segfault!
		programId = glCreateProgram();
		glUseProgram(programId);

		// Initially, we have zero shaders attached to the program
		shaderCount = 0;
	}

	// Destructor
	~ShaderProgram()
	{
		// Delete the shader program from the graphics card memory to
		// free all the resources it's been using
		glDeleteProgram(programId);
	}

	// Method to enable the shader program - we'll suggest this for inlining
	inline void use()
	{
		// Santity check that we're initialised and ready to go...
		if (initialised)
		{
			glUseProgram(programId);
		}
		else
		{
			std::string msg = "Shader program " + programId;
			msg += " not initialised - aborting.";
			throw std::runtime_error(msg);
		}
	}

	// Method to disable the shader - we'll also suggest this for inlining
	inline void disable()
	{
		glUseProgram(0);
	}

	// Method to return the bound location of a named attribute, or -1 if the attribute was not found
	GLuint attribute(const std::string attributeName)
	{
		// You could do this method with the single line:
		//
		//		return attributeMap[attribute];
		//
		// BUT, if you did, and you asked it for a named attribute which didn't exist
		// like: attributeMap["FakeAttrib"] then the method would return an invalid
		// value which will likely cause the program to segfault. So we're making sure
		// the attribute asked for exists, and if it doesn't then we alert the user & bail.

		// Create an iterator to look through our attribute map (only create iterator on first run -
		// reuse it for all further calls).
		static std::map<std::string, int>::const_iterator attributeIter;

		// Try to find the named attribute
		attributeIter = attributeMap.find(attributeName);

		// Not found? Bail.
		if (attributeIter == attributeMap.end())
		{
			throw std::runtime_error("Could not find attribute in shader program: " + attributeName);
		}

		// Otherwise return the attribute location from the attribute map
		return attributeMap[attributeName];
	}

	// Method to returns the bound location of a named uniform
	GLuint uniform(const std::string uniformName)
	{
		// Note: You could do this method with the single line:
		//
		// 		return uniformLocList[uniform];
		//
		// But we're not doing that. Explanation in the attribute() method above.

		// Create an iterator to look through our uniform map (only create iterator on first run -
		// reuse it for all further calls).
		static std::map<std::string, int>::const_iterator uniformIter;

		// Try to find the named uniform
		uniformIter = uniformMap.find(uniformName);

		// Found it? Great - pass it back! Didn't find it? Alert user and halt.
		if (uniformIter == uniformMap.end())
		{
			throw std::runtime_error("Could not find uniform in shader program: " + uniformName);
		}

		// Otherwise return the attribute location from the uniform map
		return uniformMap[uniformName];
	}

	// Method to add an attribute to the shader and return the bound location
	int addAttribute(const std::string attributeName)
	{
		// Add the attribute location value for the attributeName key
		attributeMap[attributeName] = glGetAttribLocation(programId, attributeName.c_str());

		// Check to ensure that the shader contains an attribute with this name
		if (attributeMap[attributeName] == -1)
		{
			throw std::runtime_error("Could not add attribute: " + attributeName + " - location returned -1.");
		}
		else // Valid attribute location? Inform user if we're in debug mode.
		{
			if (DEBUG)
			{
				std::cout << "Attribute " << attributeName << " bound to location: " << attributeMap[attributeName] << std::endl;
			}
		}

		// Return the attribute location
		return attributeMap[attributeName];
	}

	// Method to add a uniform to the shader and return the bound location
	int addUniform(const std::string uniformName)
	{
		// Add the uniform location value for the uniformName key
		uniformMap[uniformName] = glGetUniformLocation(programId, uniformName.c_str());

		// Check to ensure that the shader contains a uniform with this name
		if (uniformMap[uniformName] == -1)
		{
			throw std::runtime_error("Could not add uniform: " + uniformName + " - location returned -1.");
		}
		else // Valid uniform location? Inform user if we're in debug mode.
		{
			if (DEBUG)
			{
				std::cout << "Uniform " << uniformName << " bound to location: " << uniformMap[uniformName] << std::endl;
			}
		}

		// Return the uniform location
		return uniformMap[uniformName];
	}

}; // End of class

#endif // SHADER_PROGRAM_HPP
