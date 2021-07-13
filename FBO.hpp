#ifndef FBO_HPP
#define FBO_HPP

#include"prereq.h"

class FBO {
	int width, height;
	GLuint fbo;
	GLuint depthTex, integerTex, rayhitTex; 
	//rayhitTex is rgb tex that stores hit position on the cube

	void init() {
		initDepthTexture();
		initFBO();
	}

	void initDepthTexture() {
		glGenTextures(1, &depthTex);
		glBindTexture(GL_TEXTURE_2D, depthTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0,
			GL_DEPTH_COMPONENT, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE,
		//	GL_COMPARE_R_TO_TEXTURE);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
		//NULL means reserve texture memory, but texels are undefined
		//You can also try GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT24 for the internal format.
		//If GL_DEPTH24_STENCIL8_EXT, go ahead and use it (GL_EXT_packed_depth_stencil)
	}

	void initFBO() {
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		//Attach
		glBindTexture(GL_TEXTURE_2D, depthTex);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTex, 0);
		//-------------------------
		//Does the GPU support current FBO configuration?
		//Before checking the configuration, you should call these 2 according to the spec.
		//At the very least, you need to call glDrawBuffer(GL_NONE)
		glDrawBuffer(GL_NONE);
		//glReadBuffer(GL_NONE);

		checkFBO();

		disable();
	}

	void checkFBO() throw() {
		GLenum status;
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		switch (status) {
		case GL_FRAMEBUFFER_COMPLETE:
			std::cout << "Good Framebuffer\n" ;
			break;
		case GL_FRAMEBUFFER_UNDEFINED:
			throw std::runtime_error("Framebuffer undefined. Usually returned if  returned if target is the default framebuffer, but the default framebuffer does not exist.");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			throw std::runtime_error("Incomplete Attachement: is returned if any of the framebuffer attachment points are framebuffer incomplete.");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			throw std::runtime_error("Incomplete Missing Attachment: is returned if the framebuffer does not have at least one image attached to it.");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			throw std::runtime_error("Incomplete Draw Buffer: is returned if the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAWBUFFERi");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			throw std::runtime_error("Incomplete Read Buffer: is returned if GL_READ_BUFFER is not GL_NONE and the value of\\"
				" GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER");
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED:
			throw std::runtime_error("Framebuffer Unsupported: is returned if the combination of internal formats of the attached images violates an implementation-dependent set of restrictions.");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			throw std::runtime_error("Incomplete Multisample");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			throw std::runtime_error("Incomplete Layer Targets");
			break;
		default:
			throw std::runtime_error("Bad Framebuffer");
		}
	}
public:

	FBO(int w, int h) {
		width = w;
		height = h;
		init();
	}

	~FBO() {
		glDeleteFramebuffers(1, &fbo);
	}

	GLuint getDepthTexID() { return depthTex; }
	GLuint getRayhitTexID() { return rayhitTex; }
	GLuint getSDFVolPtrTexID() { return integerTex; }

	void enable() {
		//std::cout << "Render to FBO: " << fbo << "\n";
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		// Bind our frame buffer for rendering
		//-------------------------
		//----and to render to it, don't forget to call
		//At the very least, you need to call glDrawBuffer(GL_NONE)
		//glDrawBuffer(GL_NONE);
		//glReadBuffer(GL_NONE);
	}

	void disable() {
		//std::cout << "Render to screen \n";
		// Finish all operations
		//glFlush();
		//-------------------------
		//If you want to render to the back buffer again, you must bind 0 AND THEN CALL glDrawBuffer(GL_BACK)
		//else GL_INVALID_OPERATION will be raised
		glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind our texture
		glDrawBuffer(GL_BACK);
		glReadBuffer(GL_BACK);
	}

	void initIntegerTexture()	{
		enable();	//We don't need to use regular textures anymore!
		//use image instead

		glGenTextures(1, &integerTex);
		glBindTexture(GL_TEXTURE_2D, integerTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		//TODO - verify and clean this
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0,
		//		GL_RED_INTEGER, GL_UNSIGNED_INT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, width, height);
		glBindTexture(GL_TEXTURE_2D, 0);//we'll be storing ivec3(blockPos) + BlockID in texture

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, integerTex, 0);
		//GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
		//glDrawBuffers(1, drawBuffers);

		//checkFBO();

		//disable();
	}

	void initRayHitTex() {
		enable();	//We don't need to use regular textures anymore!
		//use image instead

		glGenTextures(1, &rayhitTex);
		glBindTexture(GL_TEXTURE_2D, rayhitTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		//TODO - verify and clean this
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB32F, width, height);
		glBindTexture(GL_TEXTURE_2D, 0);//we'll be storing x,y,z pos of rayhit on cube surface in texture

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rayhitTex, 0);
		//GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT1};
		//glDrawBuffers(1, drawBuffers);

		//checkFBO();

		//disable();
	}

	void initResources() {
		initIntegerTexture();
		initRayHitTex();

		//0 = blockPtr, 1 = rayHit
		GLenum drawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
		glDrawBuffers(2, drawBuffers);

		checkFBO();

		disable();
	}

};
#endif	//FBO_HPP
