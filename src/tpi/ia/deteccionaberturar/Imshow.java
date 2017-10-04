package tpi.ia.deteccionaberturar;



import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Imshow {

	public JFrame Window;
	private ImageIcon image;
	private JLabel label;
	//private MatOfByte matOfByte;
	private Boolean SizeCustom;
	private int Height, Width;

	public Imshow(String title) {
		Window = new JFrame();
		image = new ImageIcon();
		label = new JLabel();
		//matOfByte = new MatOfByte();
		label.setIcon(image);
		Window.getContentPane().add(label);
		Window.setResizable(false);
		Window.setTitle(title);
		SizeCustom = false;
		setCloseOption(0);
	}

	public Imshow(String title, int height, int width) {
		SizeCustom = true;
		Height = height;
		Width = width;

		Window = new JFrame();
		image = new ImageIcon();
		label = new JLabel();
		//matOfByte = new MatOfByte();
		label.setIcon(image);
		Window.getContentPane().add(label);
		Window.setResizable(false);
		Window.setTitle(title);
		setCloseOption(0);

	}

	public void showImage(Mat img) {
		if (SizeCustom) {
			Imgproc.resize(img, img, new Size(Height, Width));
		}
		//Highgui.imencode(".jpg", img, matOfByte);
		// byte[] byteArray = matOfByte.toArray();
		BufferedImage bufImage = null;
		try {
			// InputStream in = new ByteArrayInputStream(byteArray);
			// bufImage = ImageIO.read(in);
			bufImage = toBufferedImage(img);
			image.setImage(bufImage);
			Window.pack();
			label.updateUI();
			Window.setVisible(true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	public BufferedImage toBufferedImage(Mat m) {
		int type = BufferedImage.TYPE_BYTE_GRAY;
		if (m.channels() > 1) {
			type = BufferedImage.TYPE_3BYTE_BGR;
		}
		int bufferSize = m.channels() * m.cols() * m.rows();
		byte[] b = new byte[bufferSize];
		m.get(0, 0, b); // get all the pixels
		BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster()
				.getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);
		return image;

	}
	
	
	public void setCloseOption(int option) {

		switch (option) {
		case 0:
			Window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
			break;
		case 1:
			Window.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
			break;

		default:
			Window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

		}

	}

}
